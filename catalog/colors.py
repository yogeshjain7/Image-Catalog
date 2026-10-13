"""Colour naming.

One function (`name_array`) maps RGB pixels to a small vocabulary of colour names.
Everything else (single-pixel lookup, region dominant colour) is built on top of it,
so there is a single source of truth for what "red" or "brown" means.
"""
from __future__ import annotations

import cv2
import numpy as np

COLOR_NAMES = [
    "none", "black", "white", "gray", "red", "orange", "yellow", "green",
    "cyan", "blue", "purple", "pink", "brown", "skin",
]
IDX = {name: i for i, name in enumerate(COLOR_NAMES)}


def name_array(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB -> HxW uint8 colour ids (index into COLOR_NAMES)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    h = hsv[..., 0] * 360.0 / 255.0
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0

    out = np.full(h.shape, IDX["red"], dtype=np.uint8)
    out[(h >= 15) & (h < 45)] = IDX["orange"]
    out[(h >= 45) & (h < 70)] = IDX["yellow"]
    out[(h >= 70) & (h < 165)] = IDX["green"]
    out[(h >= 165) & (h < 200)] = IDX["cyan"]
    out[(h >= 200) & (h < 260)] = IDX["blue"]
    out[(h >= 260) & (h < 300)] = IDX["purple"]
    out[(h >= 300) & (h < 345)] = IDX["pink"]

    warm = (h < 45) | (h >= 345)
    out[warm & (v < 0.62) & (s >= 0.35)] = IDX["brown"]
    out[((h < 10) | (h >= 300)) & (s < 0.5) & (v > 0.8)] = IDX["pink"]
    out[(h >= 10) & (h <= 40) & (s >= 0.15) & (s <= 0.55) & (v >= 0.6)] = IDX["skin"]

    out[(s < 0.2) & (v < 0.5)] = IDX["gray"]
    out[s < 0.12] = np.where(v[s < 0.12] >= 0.85, IDX["white"], IDX["gray"])
    out[v < 0.18] = IDX["black"]
    return out


def name_rgb(rgb: tuple[int, int, int]) -> str:
    px = np.array([[rgb]], dtype=np.uint8)
    return COLOR_NAMES[int(name_array(px)[0, 0])]


def dominant_color(rgb: np.ndarray, mask: np.ndarray) -> str:
    """Most common colour name among the masked pixels."""
    ids = name_array(rgb)[mask.astype(bool)]
    ids = ids[ids != IDX["none"]]
    if ids.size == 0:
        return "unknown"
    return COLOR_NAMES[int(np.bincount(ids).argmax())]
