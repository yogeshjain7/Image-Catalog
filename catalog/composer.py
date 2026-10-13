"""Composite image generation from catalog assets.

    "Create an image containing one red apple and one orange"
         |
         v  1. PARSE      rule-based: counts, colour/size adjectives, nouns, layout words
    Plan(items, layout)
         |
         v  2. RESOLVE    every noun must map to a REAL catalog asset (an object crop or an image).
    Assets                 Nothing is generated or approximated; if an asset is missing we raise
         |                 MissingAssetError that names what is missing and what the catalog has.
         v  3. CUT OUT    remove the background using the same foreground model as the detector
    RGBA cutouts
         |
         v  4. LAYOUT     row / column / grid / overlap; scale + z-order; fit to canvas
    Composite PNG
         |
         v  5. REGISTER   saved as a NEW catalog entry (kind=composite) with provenance and
                          exact per-object boxes (source = "composed")

The parser is deliberately simple and deterministic: it is predictable, testable and offline.
`Composer.parser` can be replaced by an LLM that emits the same `Plan` structure (see README).
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes

from .colors import COLOR_NAMES
from .detector import ClassicalDetector
from .service import CatalogService, slugify

COUNT_WORDS = {"a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
               "pair": 2}
COLOR_WORDS = set(COLOR_NAMES) - {"none", "skin"}
SIZE_WORDS = {"tiny": 0.45, "small": 0.65, "little": 0.65, "medium": 1.0, "large": 1.35, "big": 1.35, "huge": 1.7}
IGNORED = {"colored", "coloured", "colorful", "colourful", "some", "few", "of", "the", "set", "group", "more"}

PREFIX = re.compile(
    r"^(?:please\s+)?(?:can you\s+)?"
    r"(?:(?:create|generate|make|build|compose|produce|draw|put|place|arrange|show)\b\s*(?:me\s+)?)?"
    r"(?:(?:a|an|the)\b\s+)?(?:new\s+)?(?:(?:composite|collage)\b\s*)?"
    r"(?:(?:image|picture|scene|poster)\b\s*)?"
    r"(?:(?:that\s+)?(?:containing|contains|with|of|showing|having|that has|featuring|including)\b\s*)?", re.I)
LAYOUT_HINTS = [
    (re.compile(r"\b(in a |as a )?grid\b"), "grid"),
    (re.compile(r"\b(in a |as a )?(column|vertical stack)\b|\bstacked\b|\bvertically\b"), "column"),
    (re.compile(r"\b(in a |as a )?row\b|\bside by side\b|\bnext to each other\b|\bhorizontally\b"), "row"),
    (re.compile(r"\boverlapping\b|\bpartially overlapping\b"), "overlap"),
]
RELATION = re.compile(
    r"^(?P<a>.+?)\s+(?:(?:to|on|at) the\s+)?(?P<rel>left of|right of|in front of|behind|on top of|above|below|"
    r"underneath|under|beneath|next to|beside)\s+(?P<b>.+)$")
SPLIT = re.compile(r",\s*(?:and\s+)?|\s+and\s+|\s*&\s*|\s+plus\s+|\s+along with\s+")


class MissingAssetError(ValueError):
    """The request needs catalog assets that do not exist. We refuse rather than approximate."""

    def __init__(self, missing: list[dict], available: list[str]):
        self.missing, self.available = missing, available
        lines = "; ".join(f"'{m['request']}': {m['reason']}" for m in missing)
        super().__init__(f"Cannot compose - missing catalog assets -> {lines}. Catalog contains: {', '.join(available) or 'nothing'}.")


class InstructionError(ValueError):
    pass


# --------------------------------------------------------------------------- plan
@dataclass
class Item:
    noun: str
    count: int = 1
    color: str | None = None
    size: float = 1.0
    raw: str = ""


@dataclass
class Plan:
    items: list[Item]
    layout: str = "row"            # row | column | grid | overlap (last item is drawn on top)
    notes: list[str] = field(default_factory=list)


def singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if re.search(r"(ses|xes|ches|shes)$", word):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _parse_chunk(chunk: str) -> Item | None:
    words = [w for w in re.findall(r"[a-z0-9']+", chunk.lower())]
    if not words:
        return None
    count = 1
    if words[0].isdigit():
        count, words = max(1, min(int(words[0]), 12)), words[1:]
    elif words[0] in COUNT_WORDS:
        count, words = COUNT_WORDS[words[0]], words[1:]
    colors = [w for w in words if w in COLOR_WORDS]
    sizes = [SIZE_WORDS[w] for w in words if w in SIZE_WORDS]
    nouns = [w for w in words if w not in COLOR_WORDS and w not in SIZE_WORDS and w not in IGNORED]
    if not nouns and colors:           # "one orange": the colour word IS the noun
        nouns, colors = [colors[-1]], colors[:-1]
    if not nouns:
        return None
    return Item(noun=" ".join(nouns), count=count, color=colors[-1] if colors else None,
                size=sizes[-1] if sizes else 1.0, raw=chunk.strip())


def parse_instruction(text: str) -> Plan:
    t = re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!")
    if not t:
        raise InstructionError("Instruction is empty.")
    t = PREFIX.sub("", t).strip()
    layout = "row"
    notes: list[str] = []
    for pat, name in LAYOUT_HINTS:
        if pat.search(t):
            layout = name
            t = pat.sub("", t).strip(" ,")
    m = RELATION.match(t)
    if m:
        a, rel, b = m["a"], m["rel"], m["b"]
        # `order` = left-to-right (or top-to-bottom) order; in an overlap layout the LAST item is drawn on top
        if rel in ("left of", "next to", "beside"):
            order, layout = [a, b], "row"
        elif rel == "right of":
            order, layout = [b, a], "row"
        elif rel in ("above", "on top of"):
            order, layout = [a, b], "column"
        elif rel in ("below", "under", "underneath", "beneath"):
            order, layout = [b, a], "column"
        elif rel == "in front of":
            order, layout = [b, a], "overlap"      # a is in front -> last
        else:  # behind
            order, layout = [a, b], "overlap"      # b is in front -> last
        notes.append(f"relation '{rel}' -> layout {layout}")
        chunks = [c for part in order for c in SPLIT.split(part) if c.strip()]
    else:
        chunks = [c for c in SPLIT.split(t) if c.strip()]
    items = [i for i in (_parse_chunk(c) for c in chunks) if i]
    if not items:
        raise InstructionError(f"Could not find any objects to place in: '{text}'. Try e.g. 'one red apple and one orange'.")
    total = sum(i.count for i in items)
    if total > 12:
        raise InstructionError("A composite can contain at most 12 objects.")
    return Plan(items, layout, notes)


# --------------------------------------------------------------------------- assets
@dataclass
class Asset:
    image_id: int
    image_name: str
    label: str
    color: str
    box: tuple[int, int, int, int]            # crop box in source image
    object_id: int | None
    members: list[dict]                       # source objects contained in the crop
    whole_image: bool
    alternatives: list[str] = field(default_factory=list)   # other catalog assets that also matched

    def describe(self) -> str:
        return f"{self.image_name}#{self.object_id}" if self.object_id else f"{self.image_name} (image)"


def _union_box(objs: list[dict], W: int, H: int) -> tuple[int, int, int, int]:
    x0, y0 = min(o["x"] for o in objs), min(o["y"] for o in objs)
    x1, y1 = max(o["x"] + o["w"] for o in objs), max(o["y"] + o["h"] for o in objs)
    return x0, y0, x1 - x0, y1 - y0


class Composer:
    def __init__(self, service: CatalogService):
        self.svc = service
        self._fg_detector = ClassicalDetector()

    # ---- resolve ---------------------------------------------------------
    def resolve(self, item: Item) -> Asset:
        names = {item.noun, singular(item.noun)}
        # 1) a detected object with that label (tight crop)
        objs = self.svc.find_objects_by_label(names)
        if item.color:
            colored = [o for o in objs if o["color"].lower() == item.color]
            if objs and not colored:
                have = sorted({f"{o['color']} {o['label']}" for o in objs})
                raise MissingAssetError([{"request": item.raw, "reason": f"no {item.color} {item.noun} in the catalog (found: {', '.join(have)})"}],
                                        self._available())
            objs = colored
        if objs:
            # ranking: an image literally named like the noun wins ("apple" image for "apple"), then confidence.
            names_by_image = {o["image_id"]: self.svc.get_image(o["image_id"])["name"] for o in objs}
            objs.sort(key=lambda o: (names_by_image[o["image_id"]] not in names, -o["confidence"], o["id"]))
            o = objs[0]
            alts = [f"{names_by_image[x['image_id']]}#{x['id']}" for x in objs[1:]]
            return Asset(o["image_id"], names_by_image[o["image_id"]], o["label"], o["color"],
                         (o["x"], o["y"], o["w"], o["h"]), o["id"], [o], False, alts)
        # 2) a whole catalog image with that name (crop to the union of its objects)
        for img in self.svc.list_images():
            keys = {img["name"].lower(), img["display_name"].lower(), slugify(img["display_name"])}
            if names & keys:
                full = self.svc.get_image(img["id"])
                members = full["objects"]
                if item.color and members and item.color not in {m["color"].lower() for m in members}:
                    raise MissingAssetError([{"request": item.raw, "reason": f"image '{img['name']}' has no {item.color} object"}], self._available())
                box = _union_box(members, img["width"], img["height"]) if members else (0, 0, img["width"], img["height"])
                colors = sorted({m["color"] for m in members}) or [""]
                return Asset(img["id"], img["name"], img["display_name"], colors[0] if len(colors) == 1 else "multicolor",
                             box, None, members, True)
        raise MissingAssetError([{"request": item.raw, "reason": f"nothing in the catalog is called '{item.noun}'"}], self._available())

    def _available(self) -> list[str]:
        out = []
        for img in self.svc.list_images(kind="original"):
            out.append(img["name"])
        labels = sorted({o["label"] for img in self.svc.list_images(kind="original") for o in self.svc.get_image(img["id"])["objects"]})
        return out + [f"obj:{l}" for l in labels]

    def resolve_all(self, plan: Plan) -> list[tuple[Item, Asset]]:
        resolved, missing = [], []
        for it in plan.items:
            try:
                resolved.append((it, self.resolve(it)))
            except MissingAssetError as e:
                missing.extend(e.missing)
        if missing:
            raise MissingAssetError(missing, self._available())
        return resolved

    # ---- cut-out ---------------------------------------------------------
    def cutout(self, asset: Asset) -> Image.Image:
        """Crop the asset and make its background transparent using the detector's foreground model."""
        src = self.svc.load_pil(asset.image_id).convert("RGBA")
        x, y, w, h = asset.box
        pad = 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(src.width, x + w + pad), min(src.height, y + h + pad)
        crop = src.crop((x0, y0, x1, y1))
        a = np.array(src)
        if (a[..., 3] < 250).mean() > 0.01:       # source already has real transparency
            return crop
        fg, mode, _ = self._fg_detector._foreground(np.ascontiguousarray(a[..., :3]), a[..., 3], [])
        if fg is None or mode == "flat":          # no background model -> keep the crop opaque
            return crop
        m = binary_fill_holes(np.pad(fg[y0:y1, x0:x1], 3))[3:-3, 3:-3]   # keep interior "holes" (eyes, highlights)
        alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)
        out = np.array(crop)
        out[..., 3] = (alpha * 255).astype(np.uint8)
        return Image.fromarray(out, "RGBA")

    # ---- layout ----------------------------------------------------------
    @staticmethod
    def layout(sizes: list[tuple[int, int]], factors: list[float], plan: Plan, canvas: tuple[int, int]):
        """Returns [(x, y, scale)] for each item. Bottom-aligned rows read like objects standing on a table."""
        W, H = canvas
        m = int(min(W, H) * 0.07)
        uw, uh = W - 2 * m, H - 2 * m
        n = len(sizes)
        gap = int(min(W, H) * 0.05)
        boxes: list[tuple[float, float, float, float]] = []   # x, y, w, h

        if plan.layout in ("row", "overlap"):
            # equal *visual weight*: scale by sqrt(area) so a wide strip does not shrink a tall figure
            target = uh * 0.5
            scales = [target * f / math.sqrt(w * h) for (w, h), f in zip(sizes, factors)]
            ws = [w * s for (w, _), s in zip(sizes, scales)]
            hs = [h * s for (_, h), s in zip(sizes, scales)]
            g = -min(ws) * 0.3 if plan.layout == "overlap" else gap
            total = sum(ws) + g * (n - 1)
            shrink = min(1.0, uw / total, uh / max(hs))
            scales = [s * shrink for s in scales]
            ws, hs, g = [v * shrink for v in ws], [v * shrink for v in hs], g * shrink
            total = sum(ws) + g * (n - 1)
            x, base = (W - total) / 2, H / 2 + max(hs) / 2
            for w_, h_ in zip(ws, hs):
                boxes.append((x, base - h_, w_, h_))
                x += w_ + g
        elif plan.layout == "column":
            slot = (uh - gap * (n - 1)) / n
            scales = [min(slot * f / h, uw / w) for (w, h), f in zip(sizes, factors)]
            ws = [w * s for (w, _), s in zip(sizes, scales)]
            hs = [h * s for (_, h), s in zip(sizes, scales)]
            total = sum(hs) + gap * (n - 1)
            y = (H - total) / 2
            for w_, h_ in zip(ws, hs):
                boxes.append(((W - w_) / 2, y, w_, h_))
                y += h_ + gap
        else:  # grid
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            cw, ch = uw / cols, uh / rows
            scales = [min(cw * 0.85 * f / w, ch * 0.85 * f / h) for (w, h), f in zip(sizes, factors)]
            for i, ((w, h), s) in enumerate(zip(sizes, scales)):
                r, c = divmod(i, cols)
                boxes.append((m + c * cw + (cw - w * s) / 2, m + r * ch + (ch - h * s) / 2, w * s, h * s))
        return [(b[0], b[1], b[2], b[3]) for b in boxes]

    # ---- main ------------------------------------------------------------
    def plan(self, instruction: str) -> tuple[Plan, list[tuple[Item, Asset]]]:
        plan = parse_instruction(instruction)
        return plan, self.resolve_all(plan)

    def compose(self, instruction: str, canvas: tuple[int, int] = (1024, 640), name: str | None = None,
                background: tuple[int, int, int] = (255, 255, 255)) -> dict:
        plan, resolved = self.plan(instruction)
        expanded: list[tuple[Item, Asset]] = [(it, a) for it, a in resolved for _ in range(it.count)]
        cutouts: dict[str, Image.Image] = {}
        for _, a in expanded:
            cutouts.setdefault(a.describe(), self.cutout(a))
        sizes = [cutouts[a.describe()].size for _, a in expanded]
        factors = [it.size for it, _ in expanded]
        boxes = self.layout(sizes, factors, plan, canvas)

        sheet = Image.new("RGBA", canvas, background + (255,))
        z_order = list(range(len(expanded)))
        if plan.layout == "overlap":                      # the item mentioned as "in front" is drawn last
            front = len(expanded) - 1
            z_order = [i for i in z_order if i != front] + [front]
        placements = []
        for i in z_order:
            it, a = expanded[i]
            bx, by, bw, bh = boxes[i]
            piece = cutouts[a.describe()].resize((max(1, round(bw)), max(1, round(bh))), Image.LANCZOS)
            sheet.alpha_composite(piece, (round(bx), round(by)))
        composed_objects, prov = [], []
        for i, (it, a) in enumerate(expanded):
            bx, by, bw, bh = boxes[i]
            bx, by, bw, bh = round(bx), round(by), round(bw), round(bh)
            cx, cy, cw, ch = a.box
            sx, sy = bw / max(1, cw), bh / max(1, ch)
            prov.append({"index": i, "source_image_id": a.image_id, "source_image": a.image_name,
                         "source_object_id": a.object_id, "source_box": list(a.box), "placed_box": [bx, by, bw, bh]})
            members = a.members if (a.whole_image and a.members) else None
            if members:    # image-level asset: carry each underlying object into the composite's coordinates
                for mo in members:
                    composed_objects.append({
                        "label": mo["label"], "color": mo["color"],
                        "x": max(0, bx + round((mo["x"] - cx) * sx)), "y": max(0, by + round((mo["y"] - cy) * sy)),
                        "w": max(1, round(mo["w"] * sx)), "h": max(1, round(mo["h"] * sy)),
                        "attributes": {"placement": i, "part_of": a.image_name, "source_image_id": a.image_id,
                                       "source_object_id": mo["id"]}})
            else:
                composed_objects.append({
                    "label": a.label, "color": a.color, "x": bx, "y": by, "w": bw, "h": bh,
                    "attributes": {"placement": i, "source_image_id": a.image_id, "source_object_id": a.object_id}})
        # clamp to canvas (rounding can push a box 1px out)
        for o in composed_objects:
            o["x"], o["y"] = max(0, min(o["x"], canvas[0] - 1)), max(0, min(o["y"], canvas[1] - 1))
            o["w"], o["h"] = min(o["w"], canvas[0] - o["x"]), min(o["h"], canvas[1] - o["y"])

        buf = io.BytesIO()
        sheet.convert("RGB").save(buf, "PNG")
        slug = name or ("composite_" + "_".join(slugify(it.noun) for it in plan.items)[:48])
        notes = list(plan.notes)
        for it, a in resolved:
            if a.alternatives:
                notes.append(f"'{it.raw}' used {a.describe()}; other matches: {', '.join(a.alternatives)}")
        composition = {"instruction": instruction, "layout": plan.layout, "canvas": list(canvas),
                       "parser_notes": notes, "placements": prov}
        entry = self.svc.add_image(buf.getvalue(), name=slug, original_filename=f"{slug}.png", kind="composite",
                                   detect=False, composition=composition)
        self.svc.add_composed_objects(entry["id"], composed_objects)
        return self.svc.get_image(entry["id"])
