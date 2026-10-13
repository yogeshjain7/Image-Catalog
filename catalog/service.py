"""Catalog service: storage, detection orchestration, and manual correction.

Design rules (these are what the assignment's "manual corrections are the source of truth" means):

  * Every object carries a `source`:
        auto      produced by the detector, untouched
        edited    produced by the detector, then changed by a user   (original kept in `auto_values`)
        manual    added by a user
        composed  placed by the composer (exact by construction)
  * Detection NEVER runs implicitly after the first upload. It runs again only when
    `run_detection()` is called, i.e. when the user presses "Re-run detection".
  * Re-run has two modes:
        replace_all   wipe everything, start from the detector's output
        keep_manual   replace only untouched `auto` objects; keep user work and do not resurrect
                      objects the user deleted (tombstones in `suppressed`)
  * If the detector crashes, existing objects are left exactly as they were.
"""
from __future__ import annotations

import io
import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import Image

from .detector import ClassicalDetector, Detection, Detector

OBJECT_EDITABLE = ("label", "color", "confidence", "x", "y", "w", "h", "attributes")
IMAGE_EDITABLE = ("display_name", "user_notes")
IMAGE_AUTO = ("id", "name", "width", "height", "kind", "detection_status", "detection_notes",
              "detector", "detected_at", "created_at", "original_filename", "error", "composition")

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,              -- auto: slug from filename (stable key used by the composer)
    display_name TEXT NOT NULL,             -- user-editable
    user_notes TEXT NOT NULL DEFAULT '',    -- user-editable
    original_filename TEXT,
    file_path TEXT NOT NULL,
    width INTEGER NOT NULL, height INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'original',  -- original | composite
    detection_status TEXT NOT NULL DEFAULT 'pending',  -- pending | detected | no_objects | failed | composed
    detection_notes TEXT NOT NULL DEFAULT '[]',
    detector TEXT, detected_at REAL, error TEXT,
    manually_modified INTEGER NOT NULL DEFAULT 0,
    suppressed TEXT NOT NULL DEFAULT '[]',  -- tombstones of deleted auto objects
    composition TEXT,                       -- JSON provenance for composites
    created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    label TEXT NOT NULL, color TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL,
    x INTEGER NOT NULL, y INTEGER NOT NULL, w INTEGER NOT NULL, h INTEGER NOT NULL,
    attributes TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL,                   -- auto | edited | manual | composed
    edited_fields TEXT NOT NULL DEFAULT '[]',
    auto_values TEXT,                       -- JSON snapshot of the detector's original values
    created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objects_image ON objects(image_id);
"""


class NotFound(LookupError):
    pass


class ValidationError(ValueError):
    pass


def iou(a: dict, b: dict) -> float:
    ax1, ay1, bx1, by1 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0, min(ax1, bx1) - max(a["x"], b["x"]))
    ih = max(0, min(ay1, by1) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "image"


class CatalogService:
    def __init__(self, data_dir: str | Path, detector: Detector | None = None):
        self.root = Path(data_dir)
        self.img_dir = self.root / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "catalog.db"
        self.detector: Detector = detector or ClassicalDetector()
        self._lock = threading.RLock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with self._lock:
                yield conn
                conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _obj(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["attributes"] = json.loads(d["attributes"])
        d["edited_fields"] = json.loads(d["edited_fields"])
        d["auto_values"] = json.loads(d["auto_values"]) if d["auto_values"] else None
        d["bbox"] = {"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"]}
        d["editable_fields"] = list(OBJECT_EDITABLE)
        return d

    def _img(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["detection_notes"] = json.loads(d["detection_notes"])
        d["composition"] = json.loads(d["composition"]) if d["composition"] else None
        d["manually_modified"] = bool(d["manually_modified"])
        d.pop("suppressed", None)
        d.pop("file_path", None)
        # make provenance explicit for clients: which fields a human may change, which are machine-written
        d["editable_fields"] = list(IMAGE_EDITABLE)
        d["auto_fields"] = list(IMAGE_AUTO)
        return d

    def _row(self, c, image_id: int) -> sqlite3.Row:
        row = c.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        if row is None:
            raise NotFound(f"image {image_id} not found")
        return row

    # ------------------------------------------------------------------ images
    def image_path(self, image_id: int) -> Path:
        with self._conn() as c:
            return Path(self._row(c, image_id)["file_path"])

    def load_pil(self, image_id: int) -> Image.Image:
        img = Image.open(self.image_path(image_id))
        img.load()
        return img

    def find_by_name(self, name: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM images WHERE name=?", (slugify(name),)).fetchone()
            return self._img(row) if row else None

    def list_images(self, kind: str | None = None) -> list[dict]:
        with self._conn() as c:
            q = "SELECT * FROM images" + (" WHERE kind=?" if kind else "") + " ORDER BY id"
            rows = c.execute(q, (kind,) if kind else ()).fetchall()
            out = []
            for r in rows:
                d = self._img(r)
                d["object_count"] = c.execute("SELECT COUNT(*) FROM objects WHERE image_id=?", (r["id"],)).fetchone()[0]
                out.append(d)
            return out

    def get_image(self, image_id: int) -> dict:
        with self._conn() as c:
            d = self._img(self._row(c, image_id))
            d["objects"] = [self._obj(r) for r in c.execute(
                "SELECT * FROM objects WHERE image_id=? ORDER BY x, y, id", (image_id,))]
            return d

    def add_image(self, data: bytes | str | Path, name: str | None = None, original_filename: str | None = None,
                  kind: str = "original", detect: bool = True, composition: dict | None = None) -> dict:
        """Store an image and (by default) run detection once."""
        if isinstance(data, (str, Path)):
            original_filename = original_filename or Path(data).name
            data = Path(data).read_bytes()
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as e:  # noqa: BLE001
            raise ValidationError(f"Not a readable image file: {e}") from e
        img = img.convert("RGBA") if "A" in img.getbands() else img.convert("RGB")

        base = slugify(name or (Path(original_filename).stem if original_filename else "image"))
        now = time.time()
        with self._conn() as c:
            slug, n = base, 1
            while c.execute("SELECT 1 FROM images WHERE name=?", (slug,)).fetchone():
                n += 1
                slug = f"{base}_{n}"
            cur = c.execute(
                "INSERT INTO images (name, display_name, original_filename, file_path, width, height, kind,"
                " composition, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (slug, slug, original_filename, "", img.width, img.height, kind,
                 json.dumps(composition) if composition else None, now, now))
            image_id = cur.lastrowid
            path = self.img_dir / f"{image_id}_{slug}.png"
            img.save(path)
            c.execute("UPDATE images SET file_path=? WHERE id=?", (str(path), image_id))
        if detect:
            self.run_detection(image_id, mode="replace_all")
        return self.get_image(image_id)

    def update_image(self, image_id: int, **fields: Any) -> dict:
        bad = set(fields) - set(IMAGE_EDITABLE)
        if bad:
            raise ValidationError(f"Not editable (machine-generated or unknown): {sorted(bad)}")
        if "display_name" in fields and not str(fields["display_name"]).strip():
            raise ValidationError("display_name cannot be empty")
        with self._conn() as c:
            self._row(c, image_id)
            for k, v in fields.items():
                c.execute(f"UPDATE images SET {k}=?, updated_at=? WHERE id=?", (str(v).strip(), time.time(), image_id))
        return self.get_image(image_id)

    def delete_image(self, image_id: int) -> None:
        with self._conn() as c:
            row = self._row(c, image_id)
            Path(row["file_path"]).unlink(missing_ok=True)
            c.execute("DELETE FROM images WHERE id=?", (image_id,))

    # ------------------------------------------------------------------ detection
    def run_detection(self, image_id: int, mode: str = "replace_all") -> dict:
        """(Re)run the detector. This is the ONLY way detection runs after upload."""
        if mode not in ("replace_all", "keep_manual"):
            raise ValidationError("mode must be 'replace_all' or 'keep_manual'")
        with self._conn() as c:
            row = self._row(c, image_id)
        try:
            result = self.detector.detect(self.load_pil(image_id))
        except Exception as e:  # noqa: BLE001 - a detector crash must never destroy human work
            with self._conn() as c:
                c.execute("UPDATE images SET detection_status='failed', error=?, detector=?, detected_at=?,"
                          " updated_at=? WHERE id=?",
                          (f"{type(e).__name__}: {e}", getattr(self.detector, "name", "?"), time.time(), time.time(), image_id))
            out = self.get_image(image_id)
            out["run"] = {"mode": mode, "added": 0, "kept": len(out["objects"]), "skipped": 0, "failed": True}
            return out

        now = time.time()
        added = skipped = 0
        with self._conn() as c:
            if mode == "replace_all":
                c.execute("DELETE FROM objects WHERE image_id=?", (image_id,))
                c.execute("UPDATE images SET suppressed='[]', manually_modified=0 WHERE id=?", (image_id,))
                keep: list[dict] = []
                suppressed: list[dict] = []
            else:
                c.execute("DELETE FROM objects WHERE image_id=? AND source='auto'", (image_id,))
                keep = [dict(r) for r in c.execute("SELECT * FROM objects WHERE image_id=?", (image_id,))]
                suppressed = json.loads(row["suppressed"])
            for det in result.detections:
                box = {"x": det.x, "y": det.y, "w": det.w, "h": det.h}
                if any(iou(box, k) >= 0.5 for k in keep + suppressed):
                    skipped += 1  # a human already decided about this region
                    continue
                c.execute(
                    "INSERT INTO objects (image_id,label,color,confidence,x,y,w,h,attributes,source,created_at,updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (image_id, det.label, det.color, det.confidence, det.x, det.y, det.w, det.h,
                     json.dumps(det.attributes), "auto", now, now))
                added += 1
            c.execute("UPDATE images SET detection_status=?, detection_notes=?, detector=?, detected_at=?, error=NULL,"
                      " updated_at=? WHERE id=?",
                      (result.status, json.dumps(result.notes), result.detector, now, now, image_id))
        out = self.get_image(image_id)
        out["run"] = {"mode": mode, "added": added, "kept": len(keep), "skipped": skipped, "failed": False}
        return out

    # ------------------------------------------------------------------ manual correction
    def _validate_box(self, c, image_id: int, x, y, w, h) -> tuple[int, int, int, int]:
        row = self._row(c, image_id)
        try:
            x, y, w, h = (int(round(float(v))) for v in (x, y, w, h))
        except (TypeError, ValueError) as e:
            raise ValidationError("x, y, w, h must be numbers") from e
        if w < 1 or h < 1:
            raise ValidationError("w and h must be at least 1 pixel")
        if x < 0 or y < 0 or x + w > row["width"] or y + h > row["height"]:
            raise ValidationError(f"Box must lie inside the image ({row['width']}x{row['height']})")
        return x, y, w, h

    @staticmethod
    def _validate_confidence(v) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError) as e:
            raise ValidationError("confidence must be a number between 0 and 1") from e
        if not 0.0 <= v <= 1.0:
            raise ValidationError("confidence must be between 0 and 1")
        return round(v, 3)

    def add_object(self, image_id: int, label: str, x, y, w, h, color: str = "", confidence: float = 1.0,
                   attributes: dict | None = None) -> dict:
        if not str(label).strip():
            raise ValidationError("label is required")
        now = time.time()
        with self._conn() as c:
            x, y, w, h = self._validate_box(c, image_id, x, y, w, h)
            cur = c.execute(
                "INSERT INTO objects (image_id,label,color,confidence,x,y,w,h,attributes,source,created_at,updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (image_id, str(label).strip(), str(color).strip(), self._validate_confidence(confidence), x, y, w, h,
                 json.dumps(attributes or {}), "manual", now, now))
            c.execute("UPDATE images SET manually_modified=1, updated_at=? WHERE id=?", (now, image_id))
            return self._obj(c.execute("SELECT * FROM objects WHERE id=?", (cur.lastrowid,)).fetchone())

    def update_object(self, object_id: int, **fields: Any) -> dict:
        bad = set(fields) - set(OBJECT_EDITABLE)
        if bad:
            raise ValidationError(f"Not editable: {sorted(bad)}. Editable: {list(OBJECT_EDITABLE)}")
        with self._conn() as c:
            row = c.execute("SELECT * FROM objects WHERE id=?", (object_id,)).fetchone()
            if row is None:
                raise NotFound(f"object {object_id} not found")
            cur = self._obj(row)
            new = {k: cur[k] for k in OBJECT_EDITABLE}
            new.update(fields)
            if {"x", "y", "w", "h"} & set(fields):
                new["x"], new["y"], new["w"], new["h"] = self._validate_box(c, cur["image_id"], new["x"], new["y"], new["w"], new["h"])
            if "label" in fields and not str(new["label"]).strip():
                raise ValidationError("label cannot be empty")
            if "confidence" in fields:
                new["confidence"] = self._validate_confidence(new["confidence"])
            if "attributes" in fields and not isinstance(new["attributes"], dict):
                raise ValidationError("attributes must be an object")

            changed = sorted(k for k in fields if new[k] != cur[k])
            if not changed:
                return cur
            source = cur["source"]
            auto_values = cur["auto_values"]
            if source in ("auto", "composed"):
                auto_values = {k: cur[k] for k in OBJECT_EDITABLE}  # remember what the machine said
                source = "edited"
            edited = sorted(set(cur["edited_fields"]) | set(changed))
            now = time.time()
            c.execute(
                "UPDATE objects SET label=?,color=?,confidence=?,x=?,y=?,w=?,h=?,attributes=?,source=?,"
                "edited_fields=?,auto_values=?,updated_at=? WHERE id=?",
                (str(new["label"]).strip(), str(new["color"]).strip(), new["confidence"], new["x"], new["y"], new["w"],
                 new["h"], json.dumps(new["attributes"]), source, json.dumps(edited),
                 json.dumps(auto_values) if auto_values else None, now, object_id))
            c.execute("UPDATE images SET manually_modified=1, updated_at=? WHERE id=?", (now, cur["image_id"]))
            return self._obj(c.execute("SELECT * FROM objects WHERE id=?", (object_id,)).fetchone())

    def delete_object(self, object_id: int) -> None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM objects WHERE id=?", (object_id,)).fetchone()
            if row is None:
                raise NotFound(f"object {object_id} not found")
            now = time.time()
            if row["source"] != "manual":
                # remember the machine's original box so a later "keep manual" re-run does not bring it back
                snap = json.loads(row["auto_values"]) if row["auto_values"] else row
                tomb = {k: snap[k] for k in ("label", "x", "y", "w", "h")}
                sup = json.loads(c.execute("SELECT suppressed FROM images WHERE id=?", (row["image_id"],)).fetchone()[0])
                sup.append(tomb)
                c.execute("UPDATE images SET suppressed=? WHERE id=?", (json.dumps(sup), row["image_id"]))
            c.execute("DELETE FROM objects WHERE id=?", (object_id,))
            c.execute("UPDATE images SET manually_modified=1, updated_at=? WHERE id=?", (now, row["image_id"]))

    def find_objects_by_label(self, labels: set[str], kind: str = "original") -> list[dict]:
        """Objects (any source) whose label matches one of `labels`, from images of the given kind."""
        if not labels:
            return []
        marks = ",".join("?" for _ in labels)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT o.* FROM objects o JOIN images i ON i.id=o.image_id "
                f"WHERE i.kind=? AND lower(o.label) IN ({marks}) ORDER BY o.confidence DESC, o.id",
                (kind, *[l.lower() for l in labels])).fetchall()
            return [self._obj(r) for r in rows]

    # used by the composer to register exact, known placements
    def add_composed_objects(self, image_id: int, items: list[dict]) -> None:
        now = time.time()
        with self._conn() as c:
            for it in items:
                c.execute(
                    "INSERT INTO objects (image_id,label,color,confidence,x,y,w,h,attributes,source,created_at,updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (image_id, it["label"], it.get("color", ""), 1.0, it["x"], it["y"], it["w"], it["h"],
                     json.dumps(it.get("attributes", {})), "composed", now, now))
            c.execute("UPDATE images SET detection_status='composed', detector='composer', detected_at=?,"
                      " updated_at=? WHERE id=?", (now, now, image_id))
