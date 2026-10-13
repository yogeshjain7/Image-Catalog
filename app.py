"""Flask API + single-page UI for the image catalog.

    python app.py            # http://127.0.0.1:5000
"""
from __future__ import annotations

import io
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from catalog.annotate import annotate
from catalog.composer import Composer, InstructionError, MissingAssetError
from catalog.samples import generate
from catalog.service import CatalogService, NotFound, ValidationError
from catalog.vision import build_detector

ROOT = Path(__file__).resolve().parent


def create_app(data_dir: str | Path | None = None, detector=None) -> Flask:
    app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
    svc = CatalogService(data_dir or os.environ.get("CATALOG_DATA", ROOT / "data"), detector=detector or build_detector())
    composer = Composer(svc)
    app.config["svc"], app.config["composer"] = svc, composer

    # ---- errors -> JSON ---------------------------------------------------
    @app.errorhandler(NotFound)
    def _nf(e):
        return jsonify(error=str(e)), 404

    @app.errorhandler(ValidationError)
    @app.errorhandler(InstructionError)
    def _bad(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(MissingAssetError)
    def _missing(e):
        return jsonify(error=str(e), missing=e.missing, available=e.available), 422

    def body() -> dict:
        return request.get_json(silent=True) or {}

    # ---- UI ---------------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/config")
    def config():
        vis = getattr(svc.detector, "vision", None) or (svc.detector if hasattr(svc.detector, "client") else None)
        return jsonify({"detector": svc.detector.name,
                        "vision_model": vis.client.name if vis is not None else None})

    # ---- images -----------------------------------------------------------
    @app.get("/api/images")
    def list_images():
        return jsonify(svc.list_images())

    @app.post("/api/images")
    def upload_image():
        f = request.files.get("file")
        if f is None:
            raise ValidationError("Send the image as multipart form field 'file'.")
        return jsonify(svc.add_image(f.read(), name=request.form.get("name") or None, original_filename=f.filename)), 201

    @app.get("/api/images/<int:image_id>")
    def get_image(image_id):
        return jsonify(svc.get_image(image_id))

    @app.get("/api/images/<int:image_id>/file")
    def image_file(image_id):
        return send_file(svc.image_path(image_id), mimetype="image/png", max_age=0)

    @app.get("/api/images/<int:image_id>/annotated.png")
    def image_annotated(image_id):
        data = svc.get_image(image_id)
        out = annotate(svc.load_pil(image_id), data["objects"])
        buf = io.BytesIO()
        out.save(buf, "PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    @app.patch("/api/images/<int:image_id>")
    def patch_image(image_id):
        return jsonify(svc.update_image(image_id, **body()))

    @app.delete("/api/images/<int:image_id>")
    def delete_image(image_id):
        svc.delete_image(image_id)
        return "", 204

    @app.post("/api/images/<int:image_id>/detect")
    def detect(image_id):
        return jsonify(svc.run_detection(image_id, mode=body().get("mode", "replace_all")))

    # ---- objects ----------------------------------------------------------
    @app.post("/api/images/<int:image_id>/objects")
    def add_object(image_id):
        b = body()
        missing = [k for k in ("label", "x", "y", "w", "h") if k not in b]
        if missing:
            raise ValidationError(f"Missing fields: {missing}")
        return jsonify(svc.add_object(image_id, b["label"], b["x"], b["y"], b["w"], b["h"], b.get("color", ""),
                                      b.get("confidence", 1.0), b.get("attributes"))), 201

    @app.patch("/api/objects/<int:object_id>")
    def patch_object(object_id):
        return jsonify(svc.update_object(object_id, **body()))

    @app.delete("/api/objects/<int:object_id>")
    def delete_object(object_id):
        svc.delete_object(object_id)
        return "", 204

    # ---- composites -------------------------------------------------------
    @app.post("/api/compose")
    def compose():
        b = body()
        instruction = str(b.get("instruction", "")).strip()
        if not instruction:
            raise ValidationError("instruction is required")
        return jsonify(composer.compose(instruction)), 201

    @app.post("/api/compose/preview")
    def compose_preview():
        plan, resolved = composer.plan(str(body().get("instruction", "")))
        return jsonify({"layout": plan.layout, "notes": plan.notes, "items": [
            {"request": it.raw, "count": it.count, "asset": a.describe(), "label": a.label, "color": a.color}
            for it, a in resolved]})

    # ---- demo data ----------------------------------------------------------
    @app.post("/api/samples/load")
    def load_samples():
        sample_dir = ROOT / "sample_images"
        if not any(sample_dir.glob("*.png")):
            generate(sample_dir)
        wanted = ["apple", "orange", "boy", "girl", "shapes", "colors", "apple_orange", "overlapping_shapes", "blank"]
        added = []
        for n in wanted:
            if svc.find_by_name(n) is None:
                added.append(svc.add_image(sample_dir / f"{n}.png")["name"])
        return jsonify({"added": added})

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
