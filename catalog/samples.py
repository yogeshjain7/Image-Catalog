"""Generate the 8 test images named in the assignment (+ 4 edge-case images).

Everything is drawn at 2x and downsampled so edges are anti-aliased like real
artwork. Library module; CLI wrapper: scripts/generate_test_images.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

S = 2  # supersampling factor
WHITE = (255, 255, 255)
SKIN = (255, 205, 170)


def canvas(w, h, bg=WHITE):
    img = Image.new("RGB", (w * S, h * S), bg)
    return img, ImageDraw.Draw(img)


def done(img):
    return img.resize((img.width // S, img.height // S), Image.LANCZOS)


def circle(d, cx, cy, r, fill):
    d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=fill)


def rect(d, x0, y0, x1, y1, fill):
    d.rectangle([x0 * S, y0 * S, x1 * S - 1, y1 * S - 1], fill=fill)


def poly(d, pts, fill):
    d.polygon([(x * S, y * S) for x, y in pts], fill=fill)


# ---- fruit -----------------------------------------------------------------
def draw_apple(d, cx, cy, r=110):
    circle(d, cx, cy, r, (200, 30, 35))
    circle(d, cx - r * 0.35, cy - r * 0.35, r * 0.18, (255, 150, 150))  # highlight
    rect(d, cx - 4, cy - r - 22, cx + 4, cy - r + 6, (110, 70, 30))      # stem
    poly(d, [(cx + 6, cy - r - 10), (cx + 50, cy - r - 30), (cx + 30, cy - r + 6)], (50, 170, 80))  # leaf


def draw_orange(d, cx, cy, r=105):
    circle(d, cx, cy, r, (255, 140, 0))
    circle(d, cx - r * 0.35, cy - r * 0.35, r * 0.16, (255, 185, 70))
    circle(d, cx, cy - r + 4, 7, (50, 170, 80))                           # calyx


def img_apple():
    img, d = canvas(400, 400)
    draw_apple(d, 200, 215)
    return done(img)


def img_orange():
    img, d = canvas(400, 400)
    draw_orange(d, 200, 200)
    return done(img)


def img_apple_orange():
    img, d = canvas(760, 380)
    draw_apple(d, 190, 205, 105)
    draw_orange(d, 560, 195, 100)
    return done(img)


# ---- cartoon people --------------------------------------------------------
def draw_boy(d, cx, top):
    circle(d, cx, top + 60, 52, SKIN)                                     # head
    d.pieslice([(cx - 54) * S, (top + 4) * S, (cx + 54) * S, (top + 116) * S], 180, 360, fill=(110, 70, 30))  # short hair
    circle(d, cx - 18, top + 62, 5, (20, 20, 20)); circle(d, cx + 18, top + 62, 5, (20, 20, 20))
    rect(d, cx - 12, top + 82, cx + 12, top + 87, (200, 30, 35))
    rect(d, cx - 12, top + 105, cx + 12, top + 122, SKIN)                  # neck
    rect(d, cx - 55, top + 118, cx + 55, top + 245, (50, 170, 80))        # shirt
    rect(d, cx - 85, top + 118, cx - 55, top + 190, (50, 170, 80))        # sleeves
    rect(d, cx + 55, top + 118, cx + 85, top + 190, (50, 170, 80))
    circle(d, cx - 70, top + 200, 14, SKIN); circle(d, cx + 70, top + 200, 14, SKIN)  # hands
    rect(d, cx - 55, top + 245, cx - 6, top + 355, (40, 70, 200))         # trousers (two legs)
    rect(d, cx + 6, top + 245, cx + 55, top + 355, (40, 70, 200))
    rect(d, cx - 62, top + 355, cx - 6, top + 375, (20, 20, 20))          # shoes
    rect(d, cx + 6, top + 355, cx + 62, top + 375, (20, 20, 20))


def draw_girl(d, cx, top):
    circle(d, cx, top + 60, 52, SKIN)
    d.pieslice([(cx - 54) * S, (top + 4) * S, (cx + 54) * S, (top + 116) * S], 180, 360, fill=(110, 70, 30))
    circle(d, cx - 68, top + 85, 22, (110, 70, 30)); circle(d, cx + 68, top + 85, 22, (110, 70, 30))  # pigtails
    circle(d, cx - 18, top + 62, 5, (20, 20, 20)); circle(d, cx + 18, top + 62, 5, (20, 20, 20))
    rect(d, cx - 12, top + 82, cx + 12, top + 87, (200, 30, 35))
    rect(d, cx - 12, top + 105, cx + 12, top + 122, SKIN)                  # neck
    rect(d, cx - 40, top + 118, cx + 40, top + 190, (150, 70, 200))       # bodice
    poly(d, [(cx - 40, top + 190), (cx + 40, top + 190), (cx + 95, top + 285), (cx - 95, top + 285)], (150, 70, 200))  # flared dress
    rect(d, cx - 28, top + 285, cx - 14, top + 355, SKIN); rect(d, cx + 14, top + 285, cx + 28, top + 355, SKIN)  # legs
    rect(d, cx - 36, top + 355, cx - 8, top + 372, (20, 20, 20)); rect(d, cx + 8, top + 355, cx + 36, top + 372, (20, 20, 20))


def img_boy():
    img, d = canvas(300, 400)
    draw_boy(d, 150, 8)
    return done(img)


def img_girl():
    img, d = canvas(300, 400)
    draw_girl(d, 150, 8)
    return done(img)


# ---- geometry --------------------------------------------------------------
def img_shapes():
    img, d = canvas(860, 260)
    circle(d, 110, 130, 80, (40, 70, 200))
    rect(d, 240, 50, 380, 190, (200, 30, 35))
    poly(d, [(430, 205), (510, 55), (590, 205)], (50, 170, 80))
    rect(d, 650, 80, 830, 180, (255, 140, 0))
    return done(img)


def img_colors():
    img, d = canvas(750, 180)
    for i, c in enumerate([(200, 30, 35), (40, 70, 200), (50, 170, 80), (255, 255, 0), (150, 70, 200)]):
        rect(d, i * 150, 0, (i + 1) * 150, 180, c)  # touching swatches, full bleed
    return done(img)


def img_colors_gap():
    img, d = canvas(780, 200)
    for i, c in enumerate([(200, 30, 35), (40, 70, 200), (50, 170, 80), (255, 255, 0), (150, 70, 200)]):
        rect(d, 20 + i * 150, 30, 20 + i * 150 + 130, 170, c)  # swatches on a white card
    return done(img)


def img_overlapping_shapes():
    img, d = canvas(520, 400)
    circle(d, 190, 200, 120, (200, 30, 35))                               # bottom layer
    rect(d, 240, 120, 400, 280, (40, 70, 200))                            # middle layer (covers part of circle)
    poly(d, [(330, 340), (400, 190), (470, 340)], (50, 170, 80))          # top layer (covers part of square)
    return done(img)


# ---- edge cases ------------------------------------------------------------
def img_blank():
    return Image.new("RGB", (400, 300), WHITE)


def img_noise_scene():
    """A busy, textured background with no clear subject (detection should give up)."""
    rng = np.random.default_rng(7)
    arr = rng.integers(90, 170, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def img_busy_background_apple():
    """Apple on a gradient 'wall' (non-uniform background -> GrabCut fallback)."""
    w, h = 400, 400
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    bg = np.stack([200 + 40 * xx, 190 - 40 * yy, 150 + 60 * xx * yy], -1).astype(np.uint8)
    img = Image.fromarray(bg)
    d = ImageDraw.Draw(img)
    circle_pts = [150, 140, 260, 250]
    d.ellipse(circle_pts, fill=(200, 30, 35))
    return img


def img_photo_like_apple():
    """A synthetic stand-in for a real photograph: textured wall, camera grain, shaded apple, soft shadow."""
    rng = np.random.default_rng(3)
    H, W = 480, 640
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wall = np.stack([150 + 40 * np.sin(xx / 90) + 25 * np.cos(yy / 70),
                     140 + 30 * np.cos(xx / 110) + 20 * np.sin(yy / 60),
                     120 + 35 * np.sin((xx + yy) / 130)], -1)
    # soft contact shadow
    sh = np.exp(-(((xx - 345) / 150) ** 2 + ((yy - 395) / 28) ** 2))
    wall *= (1 - 0.45 * sh)[..., None]
    # apple: radial shading, specular highlight, slight colour variation
    cx, cy, r = 330, 255, 125
    d = np.hypot(xx - cx, yy - cy)
    inside = d < r
    light = np.clip(1.0 - np.hypot(xx - (cx - 40), yy - (cy - 50)) / (1.5 * r), 0.15, 1.0)
    base = np.array([205, 28, 35], np.float32)
    apple = base[None, None, :] * (0.35 + 0.75 * light)[..., None]
    apple[..., 1] += 25 * light ** 3
    spec = np.exp(-(((xx - (cx - 50)) / 22) ** 2 + ((yy - (cy - 62)) / 14) ** 2))
    apple += 170 * spec[..., None]
    img = np.where(inside[..., None], apple, wall)
    img[(np.abs(xx - cx) < 5) & (yy > cy - r - 30) & (yy < cy - r + 12)] = (90, 60, 30)   # stem
    img += rng.normal(0, 7, img.shape)                                                       # sensor grain
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


ALL = {
    "apple": img_apple, "orange": img_orange, "boy": img_boy, "girl": img_girl,
    "shapes": img_shapes, "colors": img_colors, "apple_orange": img_apple_orange,
    "overlapping_shapes": img_overlapping_shapes,
}
EDGE = {"photo_like_apple": img_photo_like_apple, "blank": img_blank, "noise_scene": img_noise_scene, "busy_background_apple": img_busy_background_apple,
        "colors_gap": img_colors_gap}


def generate(out_dir: str | Path, include_edge_cases: bool = True) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    items = dict(ALL)
    if include_edge_cases:
        items.update(EDGE)
    for name, fn in items.items():
        p = out / f"{name}.png"
        fn().save(p)
        made.append(p)
    return made
