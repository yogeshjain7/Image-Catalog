"""Walks through the assignment's 7 test scenarios and writes evidence to demo_output/.

    python scripts/demo.py

Outputs: demo_output/annotated/*.png (boxes coloured by who made them), demo_output/composites/*.png,
         demo_output/report.md (everything printed below).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from catalog.annotate import annotate  # noqa: E402
from catalog.composer import Composer, MissingAssetError  # noqa: E402
from catalog.samples import generate  # noqa: E402
from catalog.service import CatalogService  # noqa: E402

OUT = ROOT / "demo_output"
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def table(objs: list[dict]) -> None:
    say("| id | label | color | conf | bbox (x,y,w,h) | source |")
    say("|---|---|---|---|---|---|")
    for o in objs:
        b = o["bbox"]
        say(f"| {o['id']} | {o['label']} | {o['color']} | {o['confidence']:.2f} | {b['x']},{b['y']},{b['w']},{b['h']} | {o['source']} |")
    if not objs:
        say("| - | (no objects) | | | | |")
    say()


def save_annotated(svc: CatalogService, image_id: int, name: str) -> None:
    d = svc.get_image(image_id)
    annotate(svc.load_pil(image_id), d["objects"]).save(OUT / "annotated" / f"{name}.png")


class CrashingDetector:
    name = "remote-model (simulated outage)"

    def detect(self, image):
        raise ConnectionError("detection service timed out")


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "annotated").mkdir(parents=True)
    (OUT / "composites").mkdir()
    samples = ROOT / "sample_images"
    generate(samples)
    svc = CatalogService(OUT / "data")
    comp = Composer(svc)

    say("# Demo report\n")
    say("## Scenario 1 - successful detection (names, colours, confidence, boxes)\n")
    ids = {}
    for n in ["apple", "orange", "boy", "girl", "shapes", "colors", "apple_orange", "overlapping_shapes"]:
        im = svc.add_image(samples / f"{n}.png")
        ids[n] = im["id"]
        say(f"**{n}** - {im['width']}x{im['height']}, status `{im['detection_status']}`, detector `{im['detector']}`\n")
        table(im["objects"])
        for o in im["objects"]:
            if o["attributes"].get("category") == "person":
                say(f"> clothing colours: {o['attributes']['clothing_colors']}, hair: {o['attributes']['hair_color']}\n")
        save_annotated(svc, im["id"], n)

    say("## Scenario 7a - overlapping objects\n")
    say("`overlapping_shapes` above: the circle is partly hidden by the square, the square by the triangle. Boxes overlap; "
        "the hidden circle is recognised from its *visible arc* (`occluded: true`).\n")

    say("## Scenario 2 - failed / missing detection, then manual add\n")
    blank = svc.add_image(samples / "blank.png")
    say(f"**blank** -> status `{blank['detection_status']}`, notes {blank['detection_notes']}\n")
    noise = svc.add_image(samples / "noise_scene.png")
    say(f"**noise_scene** -> status `{noise['detection_status']}`, notes {noise['detection_notes']}\n")
    added = svc.add_object(blank["id"], "pencil", 60, 50, 140, 90, color="gray", attributes={"note": "drawn by hand"})
    say("User draws a box on the blank image:\n")
    table([added])
    save_annotated(svc, blank["id"], "blank_after_manual_add")

    say("## Scenario 3 - edit label / colour / confidence / bounding box\n")
    wall = svc.add_image(samples / "busy_background_apple.png")
    say("**busy_background_apple** (gradient wall). The detector sees a red circle, not an apple (no leaf/stem to go on):\n")
    table(wall["objects"])
    o = wall["objects"][0]
    fixed = svc.update_object(o["id"], label="apple", confidence=0.95, x=148, y=138, w=116, h=116)
    say("User corrects label, confidence and box. Source becomes `edited`; the detector's answer is kept for audit:\n")
    table([fixed])
    say(f"Detector originally said: `{fixed['auto_values']}`; fields edited: `{fixed['edited_fields']}`\n")
    save_annotated(svc, wall["id"], "busy_background_apple_corrected")

    say("## Scenario 4 - corrections persist until detection is explicitly re-run\n")
    sh = svc.get_image(ids["shapes"])
    circle, square = sh["objects"][0], sh["objects"][1]
    svc.update_object(circle["id"], label="blue ball", color="blue")
    svc.add_object(ids["shapes"], "ruler", 5, 5, 40, 12)
    svc.delete_object(square["id"])
    say("Edited the circle, added a ruler, deleted the square. Re-reading the image does NOT re-detect:\n")
    table(svc.get_image(ids["shapes"])["objects"])
    r = svc.run_detection(ids["shapes"], mode="keep_manual")
    say(f"Re-run with **keep my edits** (added {r['run']['added']}, kept {r['run']['kept']}, skipped {r['run']['skipped']}) - my work survives, deleted square is not resurrected:\n")
    table(r["objects"])
    save_annotated(svc, ids["shapes"], "shapes_after_keep_manual")
    r = svc.run_detection(ids["shapes"], mode="replace_all")
    say("Re-run with **replace everything** - back to pure detector output:\n")
    table(r["objects"])

    say("## Scenario 2b - detector outage must not destroy corrections\n")
    good = svc.detector
    svc.detector = CrashingDetector()
    svc.add_object(ids["apple"], "stem", 190, 80, 15, 30)
    r = svc.run_detection(ids["apple"], mode="replace_all")
    say(f"status `{r['detection_status']}`, error `{r['error']}`; objects kept: {[o['label'] for o in r['objects']]}\n")
    svc.detector = good

    say("## Scenario 5 - composite from the existing `apple` and `orange` assets\n")
    c1 = comp.compose("Create an image containing one red apple and one orange")
    say(f"Saved as new catalog entry `{c1['name']}` (id {c1['id']}, kind `{c1['kind']}`), layout `{c1['composition']['layout']}`.\n")
    table(c1["objects"])
    say("Provenance: " + ", ".join(f"{p['source_image']}{p['source_box']} -> {p['placed_box']}" for p in c1["composition"]["placements"]) + "\n")
    shutil.copy(svc.image_path(c1["id"]), OUT / "composites" / "apple_orange.png")
    save_annotated(svc, c1["id"], "composite_apple_orange")

    say("## Scenario 6 - composite with a boy, a girl and colored shapes\n")
    c2 = comp.compose("Generate a composite containing a boy, a girl, and colored shapes")
    table(c2["objects"])
    shutil.copy(svc.image_path(c2["id"]), OUT / "composites" / "boy_girl_shapes.png")
    save_annotated(svc, c2["id"], "composite_boy_girl_shapes")
    for text, name in [("a boy in front of a girl", "boy_in_front_of_girl"),
                       ("two large apples above a small orange", "apples_above_orange")]:
        c = comp.compose(text)
        shutil.copy(svc.image_path(c["id"]), OUT / "composites" / f"{name}.png")
        say(f"Extra: `{text}` -> layout `{c['composition']['layout']}`, {len(c['objects'])} objects\n")

    say("### Required assets are never silently replaced\n")
    for text in ["one red apple and a banana", "a blue apple"]:
        try:
            comp.compose(text)
        except MissingAssetError as e:
            say(f"`{text}` -> **refused**: {e.missing}\n")
    say(f"Catalog entries after the refusals: {len(svc.list_images())} (nothing was created for them).\n")

    (OUT / "report.md").write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nWrote {OUT/'report.md'}, {len(list((OUT/'annotated').glob('*.png')))} annotated images, composites in {OUT/'composites'}")


if __name__ == "__main__":
    main()
