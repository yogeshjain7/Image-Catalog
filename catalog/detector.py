"""Object detection for flat (cartoon / illustrated) and clean photographic images.

Pipeline (all classical computer vision, runs offline on CPU in milliseconds):

  1. FOREGROUND   separate subject from background
                  - alpha channel if the image has transparency
                  - else distance from the (uniform) border colour in Lab space
                  - else "flat artwork" mode (palette image with no background, e.g. swatches)
                  - else GrabCut fallback for non-uniform backgrounds (low confidence)
  2. REGIONS      name every foreground pixel with a colour (catalog.colors) and take
                  connected components per colour -> colour regions
  3. GROUPING     foreground connected components -> groups of touching regions. For each group:
                    person  -> head (skin) on top of a tall body       -> ONE object
                    parts   -> one dominant region + small parts      -> ONE object (apple + leaf)
                    split   -> comparable regions                      -> one object PER region
  4. LABELLING    shape analysis (circle / square / rectangle / triangle, robust to
                  occlusion) + a small label lexicon (round red + leaf -> apple, ...)
  5. CONFIDENCE   heuristic score from fit quality, contrast, occlusion (NOT a calibrated
                  probability - see README for how it is used and its limits)

The `Detector` protocol at the bottom is the extension point: a YOLO/ONNX model or a hosted
vision model can be dropped in without touching the rest of the system.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .colors import COLOR_NAMES, IDX, dominant_color, name_array

MAX_SIDE = 640  # images are analysed at <= this size; boxes are scaled back


# --------------------------------------------------------------------------- types
@dataclass
class Detection:
    label: str
    color: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    attributes: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    status: str                      # "detected" | "no_objects"
    detections: list[Detection]
    notes: list[str]
    detector: str


class Detector(Protocol):
    name: str

    def detect(self, image: Image.Image) -> DetectionResult: ...


@dataclass(eq=False)  # identity comparison: masks are numpy arrays
class _Region:
    color: str
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]
    cx: float
    cy: float


@dataclass
class _Shape:
    label: str
    fit: float
    occluded: bool = False


def palette_coverage(pixels: np.ndarray, top: int = 16) -> float:
    """Share of pixels covered by the `top` most common colours (4 bits/channel).
    Flat artwork ~1.0 (a few exact colours); photographs are much lower (gradients, grain, texture)."""
    q = (pixels.reshape(-1, 3) >> 4).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    return float(np.sort(np.bincount(keys))[::-1][:top].sum() / max(1, keys.size))


PHOTO_COVERAGE = 0.80   # below this a set of pixels is treated as photographic


# --------------------------------------------------------------------------- shape analysis
def _complete_shape(c: np.ndarray, area: float, peri: float) -> _Shape:
    circ = 4 * math.pi * area / (peri * peri + 1e-9)
    if circ >= 0.85 and len(c) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(c)
        ratio = max(ma, mi) / max(1e-6, min(ma, mi))
        fit = 0.5 + 0.5 * float(np.clip((circ - 0.8) / 0.15, 0, 1))
        return _Shape("circle" if ratio < 1.15 else "ellipse", fit)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    n = len(approx)
    if n == 3:
        return _Shape("triangle", 0.9)
    if n == 4 and cv2.isContourConvex(approx):
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        rectfit = area / max(1.0, rw * rh)
        if rectfit >= 0.88:
            fit = 0.5 + 0.5 * float(np.clip((rectfit - 0.8) / 0.2, 0, 1))
            ratio = max(rw, rh) / max(1.0, min(rw, rh))
            return _Shape("square" if ratio <= 1.1 else "rectangle", fit)
        return _Shape("quadrilateral", 0.5)
    return _Shape("irregular shape", 0.35)


def _occluded_shape(c: np.ndarray, others: np.ndarray, bbox_wh: tuple[int, int]) -> _Shape:
    """Classify a shape that is partly hidden, using only the boundary the viewer can *see*
    (boundary points that are not touching another object)."""
    pts = c[:, 0, :]
    near_other = cv2.dilate(others.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    free = ~near_other[pts[:, 1], pts[:, 0]]
    if free.sum() < 20:
        return _Shape("irregular shape", 0.3, True)
    fp = pts[free].astype(np.float64)

    # (a) circle: least-squares circle through visible arc; must cover a real arc, not a straight edge
    A = np.c_[2 * fp[:, 0], 2 * fp[:, 1], np.ones(len(fp))]
    b = (fp ** 2).sum(1)
    cx, cy, k = np.linalg.lstsq(A, b, rcond=None)[0]
    r = math.sqrt(max(k + cx * cx + cy * cy, 1e-9))
    rms = float(np.sqrt(np.mean((np.hypot(fp[:, 0] - cx, fp[:, 1] - cy) - r) ** 2)))
    ang = np.sort(np.arctan2(fp[:, 1] - cy, fp[:, 0] - cx))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    coverage = 2 * math.pi - gaps.max()
    if rms / r < 0.04 and coverage > math.radians(100) and r < 2 * max(bbox_wh):
        return _Shape("circle", 0.65, True)

    # (b) polygon: dominant straight-edge directions along the visible boundary
    n = len(pts)
    idx = np.nonzero(free)[0]
    angles = []
    for i in idx:
        a, bb = pts[(i - 4) % n], pts[(i + 4) % n]
        if free[(i - 4) % n] and free[(i + 4) % n]:
            v = bb - a
            angles.append((math.degrees(math.atan2(v[1], v[0])) % 180.0))
    if len(angles) < 15:
        return _Shape("irregular shape", 0.3, True)
    hist, _ = np.histogram(angles, bins=18, range=(0, 180))
    strong = hist >= 0.10 * len(angles)
    # merge circularly-adjacent strong bins into clusters
    clusters, centres = 0, []
    for i in range(18):
        if strong[i] and not strong[i - 1]:
            j = i
            while strong[(j + 1) % 18] and (j + 1) % 18 != i:
                j += 1
            clusters += 1
            centres.append((i * 10 + (j - i + 1) * 5) % 180)
        elif strong[i] and strong.all():
            clusters = 1
    if clusters == 2:
        d = abs(centres[0] - centres[1]) % 180
        d = min(d, 180 - d)
        if abs(d - 90) <= 20:
            w, h = bbox_wh
            return _Shape("square" if 0.85 <= w / max(h, 1) <= 1.18 else "rectangle", 0.6, True)
    if clusters == 3:
        return _Shape("triangle", 0.6, True)
    return _Shape("irregular shape", 0.3, True)


def analyze_shape(mask: np.ndarray, others: np.ndarray) -> _Shape:
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return _Shape("irregular shape", 0.2)
    c = max(cnts, key=cv2.contourArea)
    area, peri = cv2.contourArea(c), cv2.arcLength(c, True)
    if area < 20:
        return _Shape("irregular shape", 0.2)
    hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-9
    solidity = area / hull_area
    touching = bool((cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool) & others).any())
    # A shape that is concave *and* touches a neighbour has a bite taken out of it.
    # The shape sitting on top (complete outline, convex) is analysed normally.
    if touching and solidity < 0.96:
        x, y, w, h = cv2.boundingRect(c)
        return _occluded_shape(c, others, (w, h))
    return _complete_shape(c, area, peri)


# --------------------------------------------------------------------------- the detector
FRUIT_LEXICON = {"red": "apple", "green": "apple", "pink": "apple", "orange": "orange", "yellow": "lemon"}


class ClassicalDetector:
    """Offline detector for flat / clean images. See module docstring."""

    name = "classical-cv-v1"

    # ---- public ----
    def detect(self, image: Image.Image) -> DetectionResult:
        rgba = np.array(image.convert("RGBA"))
        H0, W0 = rgba.shape[:2]
        scale = min(1.0, MAX_SIDE / max(H0, W0))
        if scale < 1.0:
            rgba = cv2.resize(rgba, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA)
        rgb, alpha = np.ascontiguousarray(rgba[..., :3]), rgba[..., 3]
        H, W = rgb.shape[:2]

        notes: list[str] = []
        fg, mode, contrast = self._foreground(rgb, alpha, notes)
        if mode == "grabcut":
            dets = self._grabcut_detection(rgb, fg, notes)
        elif fg is None or fg.sum() < 0.0005 * H * W:
            dets = []
        else:
            dets = self._detect_regions(rgb, fg, contrast, mode, notes)

        for d in dets:  # back to original resolution
            if scale < 1.0:
                x1, y1 = (d.x + d.w) / scale, (d.y + d.h) / scale
                d.x, d.y = int(round(d.x / scale)), int(round(d.y / scale))
                d.w, d.h = int(round(x1 - d.x)), int(round(y1 - d.y))
            d.x, d.y = max(0, d.x), max(0, d.y)
            d.w, d.h = min(d.w, W0 - d.x), min(d.h, H0 - d.y)
        if any(d.attributes.get("method") == "merged_photo_regions" for d in dets):
            notes.append("Multi-coloured photographic object: merged into one generic box. "
                         "A vision model is needed to name it (see README).")
        if not dets:
            notes.append("No detectable objects found.")
        return DetectionResult("detected" if dets else "no_objects", dets, notes, self.name)

    # ---- step 1: foreground ----
    def _foreground(self, rgb, alpha, notes):
        H, W = alpha.shape
        if (alpha < 250).mean() > 0.01:
            fg = alpha > 128
            return fg, "alpha", 60.0

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        b = 3
        border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                                 lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        if np.percentile(np.linalg.norm(border - bg, axis=1), 95) < 12:
            dist = np.linalg.norm(lab - bg, axis=2)
            fg = (dist > 22).astype(np.uint8)
            k = np.ones((3, 3), np.uint8)
            fg = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k).astype(bool)
            if fg.mean() < 0.92:
                contrast = float(np.median(dist[fg])) if fg.any() else 0.0
                return fg, "uniform_bg", contrast

        # smooth gradient background (wall, vignette, studio sweep): model it as a quadratic surface
        fit = self._smooth_background(lab)
        if fit is not None:
            dist = np.linalg.norm(lab - fit, axis=2)
            fg = (dist > 22).astype(np.uint8)
            k = np.ones((3, 3), np.uint8)
            fg = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k).astype(bool)
            if fg.mean() < 0.92:
                notes.append("Smooth gradient background modelled and removed.")
                return fg, "smooth_bg", float(np.median(dist[fg])) if fg.any() else 0.0

        # no clean background: is it flat artwork that fills the whole frame (e.g. swatches)?
        q = (rgb >> 4).reshape(-1, 3)
        keys = q[:, 0].astype(np.int32) * 256 + q[:, 1].astype(np.int32) * 16 + q[:, 2]
        counts = np.sort(np.bincount(keys))[::-1]
        if counts[:8].sum() >= 0.93 * keys.size:
            notes.append("Flat artwork fills the frame; split by colour region.")
            return np.ones((H, W), bool), "flat", 60.0

        notes.append("Background is not uniform; using GrabCut fallback (low confidence).")
        return None, "grabcut", 0.0

    @staticmethod
    def _smooth_background(lab):
        """Robustly fit a quadratic surface to the image border; None if the border is not smooth."""
        H, W, _ = lab.shape
        b = max(3, int(0.04 * min(H, W)))
        band = np.zeros((H, W), bool)
        band[:b] = band[-b:] = True
        band[:, :b] = band[:, -b:] = True
        ys, xs = np.mgrid[0:H, 0:W]
        X, Y = xs / W - 0.5, ys / H - 0.5
        feats = np.stack([np.ones_like(X), X, Y, X * Y, X * X, Y * Y], -1)
        A, target = feats[band], lab[band]
        keep = np.ones(len(A), bool)
        for _ in range(3):  # iteratively drop outliers (e.g. a subject touching the border)
            coef = np.linalg.lstsq(A[keep], target[keep], rcond=None)[0]
            res = np.linalg.norm(A @ coef - target, axis=1)
            keep = res < max(6.0, 2.5 * float(np.sqrt(np.mean(res[keep] ** 2))))
            if keep.mean() < 0.5:
                return None
        if keep.mean() < 0.9 or np.percentile(res[keep], 95) > 8:
            return None
        return feats @ coef

    def _grabcut_detection(self, rgb, _fg, notes):
        H, W = rgb.shape[:2]
        mask = np.zeros((H, W), np.uint8)
        rect = (int(W * 0.05), int(H * 0.05), int(W * 0.9), int(H * 0.9))
        try:
            cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, rect,
                        np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return []
        fg = ((mask == 1) | (mask == 3)).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n < 2:
            return []
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = lab == i
        frac = m.mean()
        labimg = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        sep = float(np.linalg.norm(labimg[m].mean(0) - labimg[~m].mean(0))) if (~m).any() else 0.0
        if not (0.03 < frac < 0.75) or sep < 25:
            notes.append("GrabCut found no clearly separated subject.")
            return []
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        notes.append("Generic label 'object' - semantic labelling needs a trained model (see README).")
        return [Detection("object", dominant_color(rgb, m), 0.40, x, y, w, h, {"method": "grabcut"})]

    # ---- steps 2-4: regions, grouping, labelling ----
    def _regions(self, rgb, fg) -> list[_Region]:
        H, W = fg.shape
        ids = np.where(fg, name_array(rgb), 0).astype(np.uint8)
        ids = cv2.medianBlur(ids, 5)
        ids[~fg] = 0
        min_area = max(25, int(0.0004 * H * W))
        k = np.ones((3, 3), np.uint8)
        out: list[_Region] = []
        for cid in np.unique(ids):
            if cid == 0:
                continue
            m = cv2.morphologyEx((ids == cid).astype(np.uint8), cv2.MORPH_OPEN, k)
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                bbox = tuple(int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
                out.append(_Region(COLOR_NAMES[cid], lab == i, area, bbox, float(cent[i][0]), float(cent[i][1])))
        return out

    def _detect_regions(self, rgb, fg, contrast, mode, notes) -> list[Detection]:
        H, W = fg.shape
        regions = self._regions(rgb, fg)
        if not regions:
            return []
        bridge = 2 * max(1, round(0.004 * max(H, W))) + 1      # bridges ~4 px gaps (neck, anti-aliasing)
        grown = cv2.dilate(fg.astype(np.uint8), np.ones((bridge, bridge), np.uint8))
        _, glab = cv2.connectedComponents(grown, connectivity=8)
        groups: dict[int, list[_Region]] = {}
        for r in regions:
            ys, xs = np.nonzero(r.mask)
            groups.setdefault(int(glab[ys[0], xs[0]]), []).append(r)

        dets: list[Detection] = []
        for gid, regs in groups.items():
            gmask = fg & (glab == gid)
            dets.extend(self._group(rgb, gmask, regs, contrast, H * W))

        # a palette of >=3 distinct rectangles reads as colour swatches
        if len(dets) >= 3 and all(d.label in ("square", "rectangle") for d in dets) and len({d.color for d in dets}) >= 3:
            for d in dets:
                d.label = f"{d.color} swatch"
        dets.sort(key=lambda d: (d.x, d.y))
        return dets

    def _group(self, rgb, gmask, regs, contrast, img_area) -> list[Detection]:
        ys, xs = np.nonzero(gmask)
        gx, gy, gw, gh = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        total = sum(r.area for r in regs)
        small_penalty = 0.8 if total < 0.003 * img_area else 1.0

        # --- person ---
        head = self._find_head(regs, gx, gy, gw, gh, total)
        if head is not None:
            return [self._person(gmask, regs, head, (gx, gy, gw, gh), total)]

        # --- a photographic, multi-coloured object (e.g. a product photo) shatters into many fragments:
        #     report ONE honest generic object instead of many confident-looking wrong shapes ---
        if len(regs) >= 2 and gmask.sum() >= 300 and palette_coverage(rgb[gmask]) < PHOTO_COVERAGE:
            main = max(regs, key=lambda r: r.area)
            return [Detection("object", main.color, 0.40, gx, gy, gw, gh,
                              {"method": "merged_photo_regions", "parts": sorted({r.color for r in regs}),
                               "fragments": len(regs)})]

        # --- one dominant region with small parts (fruit + leaf) ---
        big = max(regs, key=lambda r: r.area)
        if len(regs) > 1 and big.area >= 0.6 * total:
            # judge the dominant region's silhouette on its convex hull so a leaf/stem notch
            # in the outline does not turn a circle into an "irregular shape"
            cnts, _ = cv2.findContours(big.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            hull_mask = np.zeros(big.mask.shape, np.uint8)
            cv2.fillConvexPoly(hull_mask, cv2.convexHull(max(cnts, key=cv2.contourArea)), 1)
            sh = analyze_shape(hull_mask.astype(bool), np.zeros_like(big.mask))
            parts = [r.color for r in regs if r is not big]
            label = sh.label
            if sh.label in ("circle", "ellipse") and big.color in FRUIT_LEXICON:
                label = FRUIT_LEXICON[big.color]
            conf = self._conf(sh.fit, contrast, small_penalty) * (0.95 if label != sh.label else 1.0)
            return [Detection(label, big.color, conf, gx, gy, gw, gh,
                              {"shape": sh.label, "parts": sorted(set(parts))})]

        # --- split: one object per colour region ---
        out = []
        for r in regs:
            if len(regs) > 1 and r.area < 0.02 * total:
                continue
            others = gmask & ~r.mask
            sh = analyze_shape(r.mask, others)
            x, y, w, h = r.bbox
            attrs = {"shape": sh.label}
            if sh.occluded:
                attrs["occluded"] = True
            out.append(Detection(sh.label, r.color, self._conf(sh.fit, contrast, small_penalty), x, y, w, h, attrs))
        return out

    @staticmethod
    def _conf(fit: float, contrast: float, small_penalty: float = 1.0) -> float:
        c = (0.55 + 0.4 * fit) * (0.85 + 0.15 * min(1.0, contrast / 40.0)) * small_penalty
        return round(float(np.clip(c, 0.05, 0.99)), 2)

    # ---- people ----
    @staticmethod
    def _find_head(regs, gx, gy, gw, gh, total):
        if len(regs) < 3 or gh < 1.2 * gw:
            return None
        skins = [r for r in regs if r.color == "skin" and r.area >= 0.02 * total]
        if not skins:
            return None
        head = min(skins, key=lambda r: r.cy)
        # head = topmost skin region: near the top of the figure and a substantial part of its width
        # (hair often covers half of it, so we do not insist on a round outline)
        if head.cy > gy + 0.35 * gh or head.bbox[2] < 0.3 * gw:
            return None
        return head

    def _person(self, gmask, regs, head, gbox, total) -> Detection:
        gx, gy, gw, gh = gbox
        hx, hy, hw, hh = head.bbox

        def row_width(frac):
            r = int(gy + frac * gh)
            cols = np.nonzero(gmask[max(0, r - 2): r + 3].any(0))[0]
            return int(cols.max() - cols.min() + 1) if cols.size else 0

        torso, lower = row_width(0.45), row_width(0.72)
        flare = lower / torso if torso else 1.0
        # hair = non-skin regions around the head: above it (it may cover the head's top half)
        # or beside it (pigtails). Face features are tiny and never win the "largest" test below.
        hair_regs = [r for r in regs if r is not head and r.color != "skin"
                     and r.cy <= hy + 0.9 * hh and abs(r.cx - head.cx) < 1.1 * hw]
        if hair_regs:
            hair_w = max(r.bbox[0] + r.bbox[2] for r in hair_regs) - min(r.bbox[0] for r in hair_regs)
            hair_color = max(hair_regs, key=lambda r: r.area).color
        else:
            hair_w, hair_color = hw, "unknown"
        cue_dress = flare >= 1.25
        cue_hair = hair_w >= 1.35 * hw
        girl = cue_dress or cue_hair
        agree = cue_dress == cue_hair
        clothing = []
        for r in sorted(regs, key=lambda r: r.cy):
            if r is head or r.color == "skin" or r in hair_regs or r.area < 0.03 * total:
                continue
            if r.color not in clothing:
                clothing.append(r.color)
        return Detection(
            "girl" if girl else "boy",
            clothing[0] if clothing else "unknown",
            0.78 if agree else 0.60,
            gx, gy, gw, gh,
            {"category": "person", "clothing_colors": clothing, "hair_color": hair_color,
             "cues": {"dress_flare": round(flare, 2), "wide_hair": cue_hair}},
        )