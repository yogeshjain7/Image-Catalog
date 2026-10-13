"""Run:  python -m unittest discover -s tests -v"""
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app import create_app  # noqa: E402
from catalog.colors import name_rgb  # noqa: E402
from catalog.composer import Composer, InstructionError, MissingAssetError, parse_instruction  # noqa: E402
from catalog.detector import ClassicalDetector  # noqa: E402
from catalog.samples import generate  # noqa: E402
from catalog.service import CatalogService, ValidationError  # noqa: E402

SAMPLES = Path(tempfile.mkdtemp())
generate(SAMPLES)


def labels(res):
    return sorted(d.label for d in res.detections)


class DetectorTests(unittest.TestCase):
    det = ClassicalDetector()

    def run_on(self, name):
        return self.det.detect(Image.open(SAMPLES / f"{name}.png"))

    def test_single_fruit(self):
        r = self.run_on("apple")
        self.assertEqual([(d.label, d.color) for d in r.detections], [("apple", "red")])
        r = self.run_on("orange")
        self.assertEqual([(d.label, d.color) for d in r.detections], [("orange", "orange")])

    def test_apple_orange_two_boxes(self):
        r = self.run_on("apple_orange")
        self.assertEqual([d.label for d in r.detections], ["apple", "orange"])
        a, o = r.detections
        self.assertLess(a.x + a.w, o.x)  # individual, non-overlapping boxes

    def test_cartoon_people(self):
        boy, girl = self.run_on("boy").detections, self.run_on("girl").detections
        self.assertEqual([d.label for d in boy], ["boy"])
        self.assertEqual([d.label for d in girl], ["girl"])
        self.assertIn("blue", boy[0].attributes["clothing_colors"])
        self.assertIn("purple", girl[0].attributes["clothing_colors"])

    def test_shapes(self):
        r = self.run_on("shapes")
        self.assertEqual(sorted((d.label, d.color) for d in r.detections),
                         [("circle", "blue"), ("rectangle", "orange"), ("square", "red"), ("triangle", "green")])

    def test_colour_swatches_touching_and_spaced(self):
        for name in ("colors", "colors_gap"):
            r = self.run_on(name)
            self.assertEqual(sorted(d.color for d in r.detections), ["blue", "green", "purple", "red", "yellow"], name)

    def test_overlapping_shapes_have_overlapping_boxes(self):
        r = self.run_on("overlapping_shapes")
        self.assertEqual(sorted(d.label for d in r.detections), ["circle", "square", "triangle"])
        by = {d.label: d for d in r.detections}

        def overlap(a, b):
            return min(a.x + a.w, b.x + b.w) > max(a.x, b.x) and min(a.y + a.h, b.y + b.h) > max(a.y, b.y)
        self.assertTrue(overlap(by["circle"], by["square"]))
        self.assertTrue(overlap(by["square"], by["triangle"]))
        self.assertTrue(by["circle"].attributes.get("occluded"))

    def test_nothing_to_detect(self):
        for name in ("blank", "noise_scene"):
            r = self.run_on(name)
            self.assertEqual(r.status, "no_objects", name)
            self.assertEqual(r.detections, [])

    def test_gradient_background(self):
        r = self.run_on("busy_background_apple")
        self.assertEqual(len(r.detections), 1)
        self.assertEqual(r.detections[0].color, "red")

    def test_colour_names(self):
        self.assertEqual(name_rgb((200, 30, 35)), "red")
        self.assertEqual(name_rgb((255, 140, 0)), "orange")
        self.assertEqual(name_rgb((139, 69, 19)), "brown")


class CrashingDetector:
    name = "crash"

    def detect(self, image):
        raise RuntimeError("model service unavailable")


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.svc = CatalogService(tempfile.mkdtemp())
        self.shapes = self.svc.add_image(SAMPLES / "shapes.png")
        self.blank = self.svc.add_image(SAMPLES / "blank.png")

    def test_metadata_is_stored(self):
        im = self.shapes
        self.assertEqual((im["width"], im["height"]), (860, 260))
        o = im["objects"][0]
        for key in ("label", "color", "confidence", "bbox", "source"):
            self.assertIn(key, o)
        self.assertEqual(o["source"], "auto")
        self.assertIn("name", im["auto_fields"])
        self.assertIn("display_name", im["editable_fields"])

    def test_failed_detection_status_and_empty(self):
        self.assertEqual(self.blank["detection_status"], "no_objects")
        self.assertEqual(self.blank["objects"], [])

    def test_manual_add_edit_delete(self):
        o = self.svc.add_object(self.blank["id"], "pencil", 50, 40, 100, 80, color="gray")
        self.assertEqual(o["source"], "manual")
        auto = self.shapes["objects"][0]
        e = self.svc.update_object(auto["id"], label="big circle", x=10, confidence=0.7)
        self.assertEqual(e["source"], "edited")
        self.assertEqual(e["auto_values"]["label"], "circle")           # machine's answer is remembered
        self.assertEqual(sorted(e["edited_fields"]), ["confidence", "label", "x"])
        self.svc.delete_object(o["id"])
        self.assertEqual(self.svc.get_image(self.blank["id"])["objects"], [])

    def test_manual_corrections_persist_until_explicit_rerun(self):
        sid = self.shapes["id"]
        objs = self.svc.get_image(sid)["objects"]
        edited = self.svc.update_object(objs[0]["id"], label="my label")
        added = self.svc.add_object(sid, "extra", 1, 1, 20, 20)
        self.svc.delete_object(objs[1]["id"])                          # user deletes the square
        # a plain read never re-detects
        again = self.svc.get_image(sid)["objects"]
        self.assertIn("my label", [o["label"] for o in again])
        # keep_manual re-run: keeps my work, does not resurrect the deleted square, refreshes the rest
        r = self.svc.run_detection(sid, mode="keep_manual")
        names = [o["label"] for o in r["objects"]]
        self.assertIn("my label", names)
        self.assertIn("extra", names)
        self.assertNotIn("square", names)
        self.assertEqual(sorted(n for n in names if n not in ("my label", "extra")), ["rectangle", "triangle"])
        self.assertEqual(len(names), len(set(o["id"] for o in r["objects"])))  # no duplicates
        # replace_all re-run: back to pure detector output
        r = self.svc.run_detection(sid, mode="replace_all")
        self.assertEqual(sorted(o["label"] for o in r["objects"]), ["circle", "rectangle", "square", "triangle"])
        self.assertFalse(r["manually_modified"])
        self.assertTrue(all(o["source"] == "auto" for o in r["objects"]))
        self.assertIsNotNone(edited and added)

    def test_detector_crash_never_destroys_work(self):
        sid = self.shapes["id"]
        self.svc.add_object(sid, "mine", 1, 1, 20, 20)
        before = {o["id"] for o in self.svc.get_image(sid)["objects"]}
        self.svc.detector = CrashingDetector()
        r = self.svc.run_detection(sid, mode="replace_all")
        self.assertEqual(r["detection_status"], "failed")
        self.assertIn("model service unavailable", r["error"])
        self.assertEqual({o["id"] for o in r["objects"]}, before)

    def test_validation(self):
        sid = self.shapes["id"]
        with self.assertRaises(ValidationError):
            self.svc.add_object(sid, "x", -5, 0, 10, 10)
        with self.assertRaises(ValidationError):
            self.svc.add_object(sid, "x", 0, 0, 10, 10_000)
        with self.assertRaises(ValidationError):
            self.svc.add_object(sid, "", 0, 0, 10, 10)
        with self.assertRaises(ValidationError):
            self.svc.update_object(self.shapes["objects"][0]["id"], confidence=3)
        with self.assertRaises(ValidationError):
            self.svc.update_object(self.shapes["objects"][0]["id"], source="manual")   # provenance is not editable
        with self.assertRaises(ValidationError):
            self.svc.update_image(sid, width=1)                                         # machine field
        with self.assertRaises(ValidationError):
            self.svc.add_image(b"definitely not an image")

    def test_display_name_editable(self):
        im = self.svc.update_image(self.shapes["id"], display_name="Basic shapes", user_notes="test")
        self.assertEqual((im["display_name"], im["name"]), ("Basic shapes", "shapes"))


class ComposerTests(unittest.TestCase):
    def setUp(self):
        self.svc = CatalogService(tempfile.mkdtemp())
        for n in ("apple", "orange", "boy", "girl", "shapes"):
            self.svc.add_image(SAMPLES / f"{n}.png")
        self.comp = Composer(self.svc)

    def test_parser(self):
        p = parse_instruction("Create an image containing one red apple and one orange")
        self.assertEqual([(i.count, i.color, i.noun) for i in p.items], [(1, "red", "apple"), (1, None, "orange")])
        p = parse_instruction("a boy, a girl, and colored shapes")
        self.assertEqual([i.noun for i in p.items], ["boy", "girl", "shapes"])
        p = parse_instruction("two large apples above a small orange")
        self.assertEqual((p.layout, p.items[0].count, p.items[0].size < 2), ("column", 2, True))
        p = parse_instruction("an orange to the right of a boy")
        self.assertEqual([i.noun for i in p.items], ["boy", "orange"])
        with self.assertRaises(InstructionError):
            parse_instruction("   ")

    def test_apple_and_orange(self):
        r = self.comp.compose("Create an image containing one red apple and one orange")
        self.assertEqual((r["kind"], r["detection_status"]), ("composite", "composed"))
        self.assertEqual([o["label"] for o in r["objects"]], ["apple", "orange"])
        self.assertTrue(all(o["source"] == "composed" for o in r["objects"]))
        a, o = r["objects"]
        self.assertLess(a["x"] + a["w"], o["x"] + 1)                              # side by side, not stacked
        prov = r["composition"]["placements"]
        self.assertEqual({p["source_image"] for p in prov}, {"apple", "orange"})  # built from catalog assets
        # pixel check: the composite really contains red (apple) and orange pixels
        img = Image.open(self.svc.image_path(r["id"])).convert("RGB")
        cols = {img.getpixel((a["x"] + a["w"] // 2, a["y"] + a["h"] // 2)), img.getpixel((o["x"] + o["w"] // 2, o["y"] + o["h"] // 2))}
        self.assertEqual(cols, {(200, 30, 35), (255, 140, 0)})
        # the new entry is a normal catalog entry with its own metadata
        self.assertIn(r["id"], [i["id"] for i in self.svc.list_images()])

    def test_boy_girl_shapes(self):
        r = self.comp.compose("Generate a composite containing a boy, a girl, and colored shapes")
        self.assertEqual([o["label"] for o in r["objects"]], ["boy", "girl", "circle", "square", "triangle", "rectangle"])
        for o in r["objects"]:                                                    # every box lies inside the canvas
            self.assertTrue(0 <= o["x"] and o["x"] + o["w"] <= r["width"] and 0 <= o["y"] and o["y"] + o["h"] <= r["height"])
        shapes = [o for o in r["objects"] if o["attributes"].get("part_of") == "shapes"]
        self.assertEqual(len(shapes), 4)

    def test_counts_and_layering(self):
        r = self.comp.compose("three apples in a grid")
        self.assertEqual([o["label"] for o in r["objects"]], ["apple"] * 3)
        r = self.comp.compose("a boy in front of a girl")
        boy, girl = (next(o for o in r["objects"] if o["label"] == n) for n in ("boy", "girl"))
        self.assertTrue(min(boy["x"] + boy["w"], girl["x"] + girl["w"]) > max(boy["x"], girl["x"]))   # overlap
        img = Image.open(self.svc.image_path(r["id"])).convert("RGB")
        px = img.getpixel((boy["x"] + boy["w"] // 2, boy["y"] + int(boy["h"] * 0.35)))                  # boy's green shirt
        self.assertEqual(px, (50, 170, 80))

    def test_missing_assets_are_never_faked(self):
        before = len(self.svc.list_images())
        with self.assertRaises(MissingAssetError) as cm:
            self.comp.compose("one red apple and a banana")
        self.assertEqual(len(cm.exception.missing), 1)
        self.assertIn("banana", cm.exception.missing[0]["request"])
        with self.assertRaises(MissingAssetError) as cm:
            self.comp.compose("a blue apple")                                      # apple exists, but only red
        self.assertIn("red apple", cm.exception.missing[0]["reason"])
        self.assertEqual(len(self.svc.list_images()), before)                     # nothing was saved


class ApiTests(unittest.TestCase):
    def test_flow(self):
        app = create_app(tempfile.mkdtemp())
        c = app.test_client()
        data = (SAMPLES / "apple_orange.png").read_bytes()
        r = c.post("/api/images", data={"file": (io.BytesIO(data), "apple_orange.png")})
        self.assertEqual(r.status_code, 201)
        iid = r.json["id"]
        oid = r.json["objects"][0]["id"]
        self.assertEqual(c.patch(f"/api/objects/{oid}", json={"label": "red apple"}).json["source"], "edited")
        self.assertEqual(c.patch(f"/api/objects/{oid}", json={"id": 5}).status_code, 400)
        self.assertEqual(c.post(f"/api/images/{iid}/detect", json={"mode": "keep_manual"}).json["run"]["kept"], 1)
        self.assertEqual(c.post(f"/api/images/{iid}/objects", json={"label": "x"}).status_code, 400)
        self.assertEqual(c.get("/api/images/999").status_code, 404)
        self.assertEqual(c.post("/api/compose", json={"instruction": "a unicorn"}).status_code, 422)
        self.assertEqual(c.get(f"/api/images/{iid}/annotated.png").status_code, 200)
        self.assertEqual(c.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
