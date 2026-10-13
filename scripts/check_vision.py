"""Check that your API key works and see what the vision detector returns.

    # Windows PowerShell:   $env:GEMINI_API_KEY="your-key"
    # Mac / Linux:          export GEMINI_API_KEY="your-key"
    python scripts/check_vision.py path/to/photo.jpg

With no argument it uses sample_images/photo_like_apple.png. Writes vision_check.png with the boxes drawn.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from catalog.annotate import annotate  # noqa: E402
from catalog.samples import generate  # noqa: E402
from catalog.vision import VisionDetector, client_from_env, is_photographic  # noqa: E402


def main():
    client = client_from_env()
    if client is None:
        sys.exit("No API key found. Set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey) "
                 "or ANTHROPIC_API_KEY, then run this again.")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_images" / "photo_like_apple.png"
    if not path.exists() and path.name == "photo_like_apple.png":
        generate(ROOT / "sample_images")
    img = Image.open(path)
    print(f"Image: {path.name} {img.size}, looks photographic: {is_photographic(img)}")
    print(f"Provider/model: {client.name}")
    res = VisionDetector(client).detect(img)
    print(f"Status: {res.status}")
    for d in res.detections:
        print(f"  {d.label:14s} {d.color:8s} conf={d.confidence:.2f} box=({d.x},{d.y},{d.w},{d.h}) "
              f"refined={d.attributes['box_refined']} model_colour={d.attributes['model_color']!r}")
    objs = [{"label": d.label, "confidence": d.confidence, "source": "auto", "x": d.x, "y": d.y, "w": d.w, "h": d.h}
            for d in res.detections]
    out = ROOT / "vision_check.png"
    annotate(img, objs).save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
