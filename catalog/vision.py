"""Vision-model detection for photographic / realistic images (hosted AI service).

Why this exists: the offline detector (catalog/detector.py) is excellent on flat, cartoon-style images but
cannot name or outline objects in a real photograph. The assignment explicitly allows hosted AI services,
so photographs go to a vision model:

    image ──> HybridDetector
                 ├─ flat / cartoon image  ──> ClassicalDetector            (offline, ~30 ms, exact)
                 └─ photograph            ──> VisionDetector
                                                 1. ask the model for objects + approximate boxes
                                                 2. TIGHTEN each box with GrabCut (models are good at *what*,
                                                    only approximately good at *where* the edges are)
                                                 3. measure the colour from the object's own pixels

Providers (pick by setting an API key; see README):
    GEMINI_API_KEY     Google Gemini (has a free tier)         GEMINI_MODEL     default gemini-3.5-flash
    ANTHROPIC_API_KEY  Anthropic Claude                        ANTHROPIC_MODEL  default claude-sonnet-5-5
    VISION_PROVIDER    "gemini" | "anthropic" to force one     DETECTOR         hybrid (default) | classical | vision

Privacy note: the image is sent to the provider you configure. Nothing is sent unless a key is set and the
image looks photographic (or DETECTOR=vision).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time

import cv2
import numpy as np
import requests
from PIL import Image

from .colors import COLOR_NAMES, dominant_color
from .detector import PHOTO_COVERAGE, ClassicalDetector, Detection, DetectionResult, palette_coverage

PROMPT = (
    "You are the object detector of an image catalog. List every distinct object in the image "
    "(it may be a photograph or an illustration). Return ONLY a JSON array, no prose, no markdown. "
    'Each item: {"label": short lowercase noun such as apple, orange, boy, girl, dog, car, circle; '
    '"color": the object\'s dominant colour as one plain word; '
    '"box_2d": [ymin, xmin, ymax, xmax] as integers normalized to 0-1000 (0,0 is the top-left corner); '
    '"confidence": number between 0 and 1}. '
    "Give a separate item for each separate instance. Do not include the background. "
    "If nothing is detectable return []."
)
MAX_SEND_SIDE = 1280


class VisionError(RuntimeError):
    """The vision service failed or returned something unusable."""


# --------------------------------------------------------------------------- providers
RETRY_STATUS = {429, 500, 502, 503, 504}     # temporary: rate limit / overloaded / gateway problems
RETRY_DELAYS = (1.5, 4.0)                    # seconds to wait before the 2nd and 3rd attempt


def _readable_error(provider: str, r) -> str:
    """'Gemini returned HTTP 503: <the provider's own message>' instead of a wall of JSON."""
    msg = ""
    try:
        err = r.json().get("error", {})
        msg = err.get("message", "") if isinstance(err, dict) else str(err)
    except Exception:  # noqa: BLE001
        pass
    return f"{provider} returned HTTP {r.status_code}: {(msg or r.text)[:240]}"


def _post_with_retry(provider: str, url: str, **kw):
    """POST, retrying a few times on temporary errors (503 'high demand', 429, network blips).
    Permanent errors (401 bad key, 404 bad model, 400) are raised immediately."""
    last = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            r = requests.post(url, **kw)
        except requests.RequestException as e:
            last = VisionError(f"Could not reach {provider}: {e}")
        else:
            if r.status_code == 200:
                return r
            last = VisionError(_readable_error(provider, r))
            if r.status_code not in RETRY_STATUS:
                raise last
        if attempt < len(RETRY_DELAYS):
            time.sleep(RETRY_DELAYS[attempt])
    raise last


class GeminiClient:
    provider = "gemini"

    def __init__(self, api_key: str, model: str | None = None, timeout: int = 60, fallbacks: list[str] | None = None):
        self.api_key, self.timeout = api_key, timeout
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        env = os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash")
        self.fallbacks = [m for m in (fallbacks if fallbacks is not None else env.split(",")) if m.strip() and m.strip() != self.model]
        self.used_model = self.model

    @property
    def name(self) -> str:
        return f"gemini:{self.used_model}"

    def _call(self, model: str, image_bytes: bytes, mime: str, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": mime, "data": base64.b64encode(image_bytes).decode()}},
                {"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        }
        r = _post_with_retry("Gemini", url, headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                             json=body, timeout=self.timeout)
        try:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as e:
            raise VisionError(f"Unexpected Gemini response: {r.text[:300]}") from e

    def ask(self, image_bytes: bytes, mime: str, prompt: str) -> str:
        """Main model first (with retries); if it stays unavailable, try the fallback model(s)."""
        errors = []
        for model in [self.model, *self.fallbacks]:
            try:
                out = self._call(model, image_bytes, mime, prompt)
                self.used_model = model
                return out
            except VisionError as e:
                errors.append(f"{model}: {e}")
        raise VisionError(" | ".join(errors))


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key: str, model: str | None = None, timeout: int = 60):
        self.api_key, self.timeout = api_key, timeout
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def ask(self, image_bytes: bytes, mime: str, prompt: str) -> str:
        body = {
            "model": self.model, "max_tokens": 2000, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime,
                                             "data": base64.b64encode(image_bytes).decode()}},
                {"type": "text", "text": prompt}]}],
        }
        r = _post_with_retry("Anthropic", "https://api.anthropic.com/v1/messages",
                             headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                                      "content-type": "application/json"},
                             json=body, timeout=self.timeout)
        try:
            return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
        except (KeyError, ValueError, TypeError) as e:
            raise VisionError(f"Unexpected Anthropic response: {r.text[:300]}") from e


def client_from_env():
    """The configured provider, or None if no API key is set."""
    forced = os.environ.get("VISION_PROVIDER", "").lower()
    gem = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ant = os.environ.get("ANTHROPIC_API_KEY")
    if forced == "anthropic" and ant:
        return AnthropicClient(ant)
    if forced == "gemini" and gem:
        return GeminiClient(gem)
    if not forced:
        if gem:
            return GeminiClient(gem)
        if ant:
            return AnthropicClient(ant)
    return None


# --------------------------------------------------------------------------- helpers
def encode_for_model(image: Image.Image) -> tuple[bytes, str]:
    """JPEG, max 1280 px on the long side, transparency flattened onto white."""
    img = image.convert("RGBA")
    flat = Image.new("RGB", img.size, (255, 255, 255))
    flat.paste(img, mask=img.split()[3])
    if max(flat.size) > MAX_SEND_SIDE:
        s = MAX_SEND_SIDE / max(flat.size)
        flat = flat.resize((round(flat.width * s), round(flat.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    flat.save(buf, "JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def is_photographic(image: Image.Image) -> bool:
    """Flat artwork is made of a handful of exact colours; a photograph is not.

    Measured on the SUBJECT only: on a product photo most pixels are plain white background, which would
    otherwise make the picture look flat."""
    img = image.convert("RGB")
    img.thumbnail((200, 200))
    a = np.asarray(img).astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    if (np.abs(border - bg).sum(1) < 30).mean() > 0.9:            # plain background: ignore it
        pixels = a[np.abs(a - bg).sum(2) > 40]
        if len(pixels) < 150:
            return False                                           # (almost) nothing there: not a photo of anything
    else:
        pixels = a.reshape(-1, 3)
    return palette_coverage(pixels.astype(np.uint8)) < PHOTO_COVERAGE


_COLOR_SYNONYMS = {"grey": "gray", "violet": "purple", "magenta": "pink", "tan": "brown", "beige": "skin",
                   "golden": "yellow", "gold": "yellow", "navy": "blue", "teal": "cyan", "turquoise": "cyan",
                   "maroon": "brown", "crimson": "red", "lime": "green"}


def normalize_color(word: str) -> str | None:
    """Map a model's colour word onto our vocabulary (None if it is not one we know)."""
    w = word.strip().lower().split()[-1] if word.strip() else ""
    w = _COLOR_SYNONYMS.get(w, w)
    return w if w in COLOR_NAMES and w != "none" else None


def _iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def parse_model_output(text: str) -> list[dict]:
    """Tolerant JSON extraction: accepts a bare array, a fenced block, or {"objects": [...]}."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    data = None
    try:
        data = json.loads(t)
    except ValueError:
        m = re.search(r"\[.*\]", t, flags=re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except ValueError:
                pass
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        raise VisionError(f"Model did not return a JSON list: {text[:200]!r}")
    items = []
    for it in data:
        if not isinstance(it, dict):
            continue
        box = it.get("box_2d") or it.get("bbox") or it.get("box")
        label = str(it.get("label") or it.get("name") or "").strip().lower()
        try:
            ymin, xmin, ymax, xmax = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        ymin, ymax = sorted((max(0.0, min(1000.0, ymin)), max(0.0, min(1000.0, ymax))))
        xmin, xmax = sorted((max(0.0, min(1000.0, xmin)), max(0.0, min(1000.0, xmax))))
        if not label or ymax - ymin < 5 or xmax - xmin < 5:
            continue
        try:
            conf = float(it.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.8
        items.append({"label": label, "color": str(it.get("color", "")).strip().lower(),
                      "norm": (ymin, xmin, ymax, xmax), "confidence": max(0.0, min(1.0, conf))})
    return items


def refine_box(rgb: np.ndarray, box: tuple[int, int, int, int]):
    """Tighten a model's approximate box with GrabCut. Returns (box, mask_in_box_coords | None, refined?)."""
    H, W = rgb.shape[:2]
    x, y, w, h = box
    pad = int(0.10 * max(w, h))
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    sub = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    if sub.shape[0] < 12 or sub.shape[1] < 12 or w < 8 or h < 8:
        return box, None, False
    s = min(1.0, 320 / max(sub.shape[:2]))
    small = cv2.resize(sub, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else sub
    rect = (max(0, int((x - x0) * s)), max(0, int((y - y0) * s)),
            max(4, int(w * s)), max(4, int(h * s)))
    rect = (rect[0], rect[1], min(rect[2], small.shape[1] - rect[0] - 1), min(rect[3], small.shape[0] - rect[1] - 1))
    mask = np.zeros(small.shape[:2], np.uint8)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, rect, np.zeros((1, 65)), np.zeros((1, 65)),
                    3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return box, None, False
    fg = ((mask == 1) | (mask == 3)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n < 2:
        return box, None, False
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    new = (x0 + int(bx / s), y0 + int(by / s), max(1, int(bw / s)), max(1, int(bh / s)))
    frac = (lab == i).sum() / max(1, rect[2] * rect[3])
    if _iou(new, box) < 0.5 or not 0.15 <= frac <= 1.1:
        return box, None, False          # GrabCut disagreed wildly with the model: trust the model
    mask_full = cv2.resize((lab == i).astype(np.uint8), (sub.shape[1], sub.shape[0]), interpolation=cv2.INTER_NEAREST)
    return new, (mask_full, (x0, y0)), True


class VisionDetector:
    """Hosted vision model for labels + approximate boxes, GrabCut for tight boxes, pixels for colour."""

    def __init__(self, client, refine: bool = True):
        self.client, self.refine = client, refine

    @property
    def name(self) -> str:
        return f"{self.client.name}{'+grabcut' if self.refine else ''}"

    def detect(self, image: Image.Image) -> DetectionResult:
        data, mime = encode_for_model(image)
        text = self.client.ask(data, mime, PROMPT)
        items = parse_model_output(text)
        rgb = np.ascontiguousarray(np.array(image.convert("RGB")))
        H, W = rgb.shape[:2]
        dets: list[Detection] = []
        for it in items:
            ymin, xmin, ymax, xmax = it["norm"]
            box = (int(xmin / 1000 * W), int(ymin / 1000 * H),
                   max(1, int((xmax - xmin) / 1000 * W)), max(1, int((ymax - ymin) / 1000 * H)))
            refined, mask, ok = refine_box(rgb, box) if self.refine else (box, None, False)
            if any(d.label == it["label"] and _iou(refined, (d.x, d.y, d.w, d.h)) > 0.85 for d in dets):
                continue                                   # duplicate of an earlier item
            x, y, w, h = refined
            x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
            w, h = min(w, W - x), min(h, H - y)
            if ok:
                m, (ox, oy) = mask
                full = np.zeros((H, W), bool)
                full[oy:oy + m.shape[0], ox:ox + m.shape[1]] = m.astype(bool)
                color = dominant_color(rgb, full)
            else:                                           # centre of the box: avoids background at the corners
                cx0, cy0 = x + w // 4, y + h // 4
                core = np.zeros((H, W), bool)
                core[cy0:cy0 + max(1, h // 2), cx0:cx0 + max(1, w // 2)] = True
                color = dominant_color(rgb, core)
            # Shading makes pixel-based colour unreliable on photographs (a shadowed red apple reads as brown),
            # so prefer the model's colour word when it is in our vocabulary; keep the measurement as evidence.
            measured, color = color, (normalize_color(it["color"]) or color)
            conf = round(min(0.95, it["confidence"]) * (1.0 if ok else 0.85), 2)
            dets.append(Detection(it["label"], color, conf, x, y, w, h,
                                  {"method": "vision", "model": self.client.name, "model_color": it["color"],
                                   "box_refined": ok}))
        dets.sort(key=lambda d: (d.x, d.y))
        notes = [f"Detected by {self.client.name}; confidence is the model's own estimate, not calibrated."]
        if not dets:
            notes.append("No detectable objects found.")
        return DetectionResult("detected" if dets else "no_objects", dets, notes, self.name)


class HybridDetector:
    """Route each image to the detector that suits it (see module docstring)."""

    name = "hybrid"

    def __init__(self, classical: ClassicalDetector | None = None, vision: VisionDetector | None = None):
        self.classical = classical or ClassicalDetector()
        self.vision = vision

    def detect(self, image: Image.Image) -> DetectionResult:
        res = self.classical.detect(image)
        used_fallback = any("GrabCut" in n for n in res.notes)      # offline detector found no clean background
        shattered = any(d.attributes.get("method") == "merged_photo_regions" for d in res.detections)
        if not (is_photographic(image) or used_fallback or shattered):
            return res                                      # flat / cartoon: the offline detector is exact
        if self.vision is None:
            res.notes.append("This looks like a photograph. The offline detector only handles clean or flat images; "
                             "set GEMINI_API_KEY or ANTHROPIC_API_KEY to enable the vision detector.")
            return res
        try:
            out = self.vision.detect(image)
        except Exception as e:  # noqa: BLE001
            if res.detections:                              # degrade gracefully, but say so
                res.notes.append(f"Vision detector unavailable ({e}); showing the offline result instead.")
                return res
            raise                                           # nothing to fall back on -> status 'failed' with the reason
        out.notes.insert(0, "Photographic image: routed to the vision model.")
        return out


def build_detector():
    """Detector chosen from the environment (used by app.py)."""
    mode = os.environ.get("DETECTOR", "hybrid").lower()
    classical = ClassicalDetector()
    if mode == "classical":
        return classical
    client = client_from_env()
    if mode == "vision":
        if client is None:
            raise RuntimeError("DETECTOR=vision needs GEMINI_API_KEY or ANTHROPIC_API_KEY")
        return VisionDetector(client)
    return HybridDetector(classical, VisionDetector(client) if client else None)