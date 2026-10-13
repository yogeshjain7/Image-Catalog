"""CLI: python scripts/generate_test_images.py [output_dir]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.samples import generate  # noqa: E402

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "sample_images")
    for p in generate(target):
        print("wrote", p)
