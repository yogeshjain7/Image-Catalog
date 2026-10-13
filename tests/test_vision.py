"""Vision/hybrid detector tests. No network and no API key needed: the HTTP layer and the model are faked."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from catalog import vision  # noqa: E402
from catalog.detector import ClassicalDetector  # noqa: E402
from catalog.samples import generate  # noqa: E402
from catalog.service import CatalogService  # noqa: E402
from catalog.vision import (AnthropicClient, GeminiClient, HybridDetector, VisionDetector, VisionError,  # noqa: E402
                            is_photographic, parse_model_output)

SAMPLES = Path(tempfile.mkdtemp())
generate(SAMPLES)
PHOTO = Image.open(SAMPLES / "photo_like_apple.png")       # 640x480, apple body truth = (205,130,250,250)
TRUTH = (205, 130, 250, 250)


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / (a[2] * a[3] + b[2] * b[3] - i)


class FakeClient:
    """Pretends to be a model: returns a deliberately sloppy box around the apple."""
    name = "fake:model"

    def __init__(self, reply=None):
        self.calls = 0
        self.reply = reply

    def ask(self, data, mime, prompt):
        self.calls += 1
        assert mime == "image/jpeg" and data[:2] == b"\xff\xd8" and "JSON" in prompt
        if self.reply is not None:
            return self.reply
        return json.dumps([{"label": "Apple", "color": "red", "confidence": 0.93,
                            "box_2d": [round(100 / 480 * 1000), round(180 / 640 * 1000),
                                       round(410 / 480 * 1000), round(480 / 640 * 1000)]}])


class ParsingTests(unittest.TestCase):
    def test_accepts_messy_model_output(self):
        for text in ['[{"label":"cat","box_2d":[10,20,300,400]}]',
                     '```json\n[{"label":"cat","box_2d":[10,20,300,400]}]\n```',
                     'Sure! Here you go: [{"label":"cat","box_2d":[10,20,300,400]}] hope it helps',
                     '{"objects":[{"label":"cat","bbox":[10,20,300,400]}]}']:
            items = parse_model_output(text)
            self.assertEqual([i["label"] for i in items], ["cat"], text)

    def test_clamps_and_drops_garbage(self):
        items = parse_model_output(json.dumps([
            {"label": "a", "box_2d": [-50, 0, 2000, 500]},        # out of range -> clamped
            {"label": "b", "box_2d": [500, 500, 500, 900]},       # zero height -> dropped
            {"label": "", "box_2d": [0, 0, 100, 100]},            # no label -> dropped
            {"label": "c", "box_2d": "oops"},                     # not a box -> dropped
            {"label": "d", "box_2d": [900, 800, 100, 200]},       # swapped corners -> fixed
            "junk"]))
        self.assertEqual([i["label"] for i in items], ["a", "d"])
        self.assertEqual(items[0]["norm"], (0.0, 0.0, 1000.0, 500.0))
        self.assertEqual(items[1]["norm"], (100.0, 200.0, 900.0, 800.0))

    def test_not_json_raises(self):
        with self.assertRaises(VisionError):
            parse_model_output("I cannot see any image.")

    def test_photo_routing_signal(self):
        self.assertTrue(is_photographic(PHOTO))
        for n in ("apple", "boy", "girl", "shapes", "colors", "apple_orange", "overlapping_shapes", "blank",
                  "busy_background_apple"):
            self.assertFalse(is_photographic(Image.open(SAMPLES / f"{n}.png")), n)


class VisionDetectorTests(unittest.TestCase):
    def test_label_colour_and_tight_box(self):
        raw = VisionDetector(FakeClient(), refine=False).detect(PHOTO).detections[0]
        res = VisionDetector(FakeClient(), refine=True).detect(PHOTO)
        d = res.detections[0]
        self.assertEqual((d.label, d.color), ("apple", "red"))                 # not the pixel-measured "brown"
        self.assertEqual(d.attributes["model_color"], "red")
        self.assertTrue(d.attributes["box_refined"])
        self.assertGreater(iou((d.x, d.y, d.w, d.h), TRUTH), iou((raw.x, raw.y, raw.w, raw.h), TRUTH))
        self.assertGreater(iou((d.x, d.y, d.w, d.h), TRUTH), 0.8)
        self.assertTrue(0 < d.confidence <= 0.95)
        self.assertEqual(res.status, "detected")

    def test_empty_answer_is_no_objects(self):
        res = VisionDetector(FakeClient(reply="[]")).detect(PHOTO)
        self.assertEqual((res.status, res.detections), ("no_objects", []))

    def test_boxes_stay_inside_image(self):
        reply = json.dumps([{"label": "thing", "box_2d": [0, 0, 1000, 1000], "confidence": 1}])
        d = VisionDetector(FakeClient(reply=reply)).detect(PHOTO).detections[0]
        self.assertTrue(d.x >= 0 and d.y >= 0 and d.x + d.w <= 640 and d.y + d.h <= 480)


class HybridTests(unittest.TestCase):
    def test_flat_images_never_call_the_model(self):
        fake = FakeClient()
        hyb = HybridDetector(ClassicalDetector(), VisionDetector(fake))
        for n in ("apple", "boy", "shapes", "overlapping_shapes", "blank"):
            hyb.detect(Image.open(SAMPLES / f"{n}.png"))
        self.assertEqual(fake.calls, 0)

    def test_photo_goes_to_the_model(self):
        fake = FakeClient()
        res = HybridDetector(ClassicalDetector(), VisionDetector(fake)).detect(PHOTO)
        self.assertEqual(fake.calls, 1)
        self.assertEqual(res.detections[0].label, "apple")
        self.assertIn("vision model", res.notes[0])

    def test_textured_scene_with_no_clean_background_also_goes_to_the_model(self):
        fake = FakeClient(reply="[]")
        res = HybridDetector(ClassicalDetector(), VisionDetector(fake)).detect(Image.open(SAMPLES / "noise_scene.png"))
        self.assertEqual(fake.calls, 1)
        self.assertEqual(res.status, "no_objects")

    def test_without_a_key_falls_back_and_says_why(self):
        res = HybridDetector(ClassicalDetector(), None).detect(PHOTO)
        self.assertTrue(any("GEMINI_API_KEY" in n for n in res.notes))

    def test_vision_outage_with_nothing_to_fall_back_on_fails_loudly(self):
        class Down:
            name = "down"

            def ask(self, *a):
                raise VisionError("HTTP 503")
        hyb = HybridDetector(ClassicalDetector(), VisionDetector(Down()))
        svc = CatalogService(tempfile.mkdtemp(), detector=hyb)
        r = svc.add_image(SAMPLES / "noise_scene.png")             # classical finds nothing -> failure is surfaced
        self.assertEqual(r["detection_status"], "failed")
        self.assertIn("HTTP 503", r["error"])

    def test_vision_outage_degrades_to_offline_result_when_it_has_one(self):
        class Down:
            name = "down"

            def ask(self, *a):
                raise VisionError("HTTP 503")
        hyb = HybridDetector(ClassicalDetector(), VisionDetector(Down()))
        res = hyb.detect(PHOTO)                                      # classical GrabCut guess exists
        self.assertTrue(res.detections)
        self.assertTrue(any("unavailable" in n for n in res.notes))

    def test_manual_edits_survive_a_vision_run(self):
        svc = CatalogService(tempfile.mkdtemp(), detector=HybridDetector(ClassicalDetector(), VisionDetector(FakeClient())))
        im = svc.add_image(SAMPLES / "photo_like_apple.png")
        self.assertEqual(im["objects"][0]["label"], "apple")
        svc.update_object(im["objects"][0]["id"], label="red delicious apple")
        r = svc.run_detection(im["id"], mode="keep_manual")
        self.assertEqual([o["label"] for o in r["objects"]], ["red delicious apple"])


class ProviderRequestTests(unittest.TestCase):
    """Check the exact HTTP requests we would send (shapes taken from the providers' documentation)."""

    def test_gemini_request_and_response(self):
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "[]"}]}}]}
        with mock.patch.object(vision.requests, "post", return_value=resp) as post:
            out = GeminiClient("KEY", model="gemini-x").ask(b"\xff\xd8abc", "image/jpeg", "prompt")
        self.assertEqual(out, "[]")
        url, kw = post.call_args.args[0], post.call_args.kwargs
        self.assertTrue(url.endswith("/v1beta/models/gemini-x:generateContent"))
        self.assertEqual(kw["headers"]["x-goog-api-key"], "KEY")
        parts = kw["json"]["contents"][0]["parts"]
        self.assertEqual(parts[0]["inline_data"]["mime_type"], "image/jpeg")
        self.assertEqual(parts[1]["text"], "prompt")
        self.assertEqual(kw["json"]["generationConfig"]["responseMimeType"], "application/json")

    def test_anthropic_request_and_response(self):
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"content": [{"type": "text", "text": "[]"}]}
        with mock.patch.object(vision.requests, "post", return_value=resp) as post:
            out = AnthropicClient("KEY", model="claude-x").ask(b"\xff\xd8abc", "image/jpeg", "prompt")
        self.assertEqual(out, "[]")
        kw = post.call_args.kwargs
        self.assertEqual(post.call_args.args[0], "https://api.anthropic.com/v1/messages")
        self.assertEqual(kw["headers"]["x-api-key"], "KEY")
        self.assertEqual(kw["headers"]["anthropic-version"], "2023-06-01")
        block = kw["json"]["messages"][0]["content"][0]
        self.assertEqual((block["type"], block["source"]["type"], block["source"]["media_type"]), ("image", "base64", "image/jpeg"))
        self.assertEqual(kw["json"]["model"], "claude-x")

    def test_http_errors_become_readable_vision_errors(self):
        resp = mock.Mock(status_code=401, text='{"error":"bad key"}')
        for client in (GeminiClient("K"), AnthropicClient("K")):
            with mock.patch.object(vision.requests, "post", return_value=resp):
                with self.assertRaises(VisionError) as cm:
                    client.ask(b"x", "image/jpeg", "p")
            self.assertIn("401", str(cm.exception))

    def test_provider_selection_from_environment(self):
        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}, clear=True):
            self.assertEqual(vision.client_from_env().provider, "gemini")
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "a"}, clear=True):
            self.assertEqual(vision.client_from_env().provider, "anthropic")
        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a", "VISION_PROVIDER": "anthropic"}, clear=True):
            self.assertEqual(vision.client_from_env().provider, "anthropic")
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(vision.client_from_env())
            self.assertIsInstance(vision.build_detector(), HybridDetector)


if __name__ == "__main__":
    unittest.main()
