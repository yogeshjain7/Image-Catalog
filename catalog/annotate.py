"""Draw detected objects on an image. Colour = who made the box (the key idea of this system)."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

SOURCE_COLORS = {
    "auto": (36, 87, 230),      # blue   - detected by the machine
    "edited": (232, 137, 12),   # amber  - machine box that a human corrected
    "manual": (232, 137, 12),   # amber  - added by a human
    "composed": (14, 159, 142), # teal   - placed by the composer
}


def annotate(image: Image.Image, objects: list[dict], show_conf: bool = True) -> Image.Image:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(11, img.height // 40))
    except OSError:
        font = ImageFont.load_default()
    lw = max(2, img.width // 300)
    for o in objects:
        col = SOURCE_COLORS.get(o["source"], (90, 90, 90))
        x, y, w, h = o["x"], o["y"], o["w"], o["h"]
        d.rectangle([x, y, x + w, y + h], outline=col, width=lw)
        tag = o["label"] + (f" {o['confidence']:.2f}" if show_conf else "")
        tw = d.textlength(tag, font=font)
        th = font.size + 4 if hasattr(font, "size") else 14
        ty = y - th if y - th >= 0 else y
        d.rectangle([x, ty, x + tw + 8, ty + th], fill=col)
        d.text((x + 4, ty + 1), tag, fill=(255, 255, 255), font=font)
    return img
