# Question 1 - Image catalog, object detection, metadata and composite generation

A small, working system that

1. keeps a **catalog** of images with metadata (SQLite + files on disk),
2. **detects objects automatically** on upload (cartoon *and* clean photographic images),
3. lets a human **add / edit / delete** objects, with manual work always winning,
4. **composes new images** from existing catalog assets using a natural-language instruction, and saves each result as a new catalog entry.

Python 3.10+, Flask, OpenCV, Pillow. Cartoon/flat images are detected offline on CPU in ~30 ms; **photographs are sent to a hosted vision model** (Gemini or Claude) when you set an API key - see section 2b.

```bash
pip install -r requirements.txt
python scripts/generate_test_images.py        # creates sample_images/ (the 8 assignment images + edge cases)
python app.py                                 # UI + API at http://127.0.0.1:5000  ("Load sample images" button)
python scripts/demo.py                        # runs all 7 scenarios, writes demo_output/report.md + annotated images
python -m unittest discover -s tests -v       # 40 tests (no API key needed)
```

---

## 1. Architecture

```
 upload ──> CatalogService ──> Detector (pluggable) ──> objects table  (source = auto)
               │  ▲                                          │
               │  └────── manual add / edit / delete ────────┘  (source = manual / edited)
               │
 "one red apple and one orange"
               │
               └─> Composer: parse -> resolve REAL assets -> cut out -> layout -> save as new catalog entry
                                                                         (objects source = composed)
```

| Module | Responsibility |
|---|---|
| `catalog/vision.py` | Photographs: hosted vision model (Gemini / Claude) + GrabCut box tightening + routing (`HybridDetector`). |
| `catalog/detector.py` | Foreground separation, colour regions, grouping, shape analysis, labelling, confidence. `Detector` protocol = extension point. |
| `catalog/colors.py` | One function that names pixels (`red`, `brown`, `skin` ...). Single source of truth for colour. |
| `catalog/service.py` | SQLite storage, provenance rules, detection orchestration, validation. |
| `catalog/composer.py` | Instruction parser, asset resolver, cut-outs, layout engine, composite registration. |
| `app.py`, `static/index.html` | REST API and single-page UI. |

### Data model

`images`: `name` (auto, stable key), **`display_name`** and **`user_notes`** (user-editable), `width`, `height`, `kind` (original / composite), `detection_status` (pending / detected / no_objects / failed / composed), `detector`, `detected_at`, `error`, `composition` (provenance JSON for composites).

`objects`: `label`, `color`, `confidence`, `x, y, w, h`, `attributes` (JSON: clothing colours, shape, parts, occluded ...), and the provenance fields **`source`**, `edited_fields`, `auto_values`.

**Auto vs editable is explicit, at two levels.** The API returns `auto_fields` / `editable_fields` for each image, and the UI shows machine-written values in a grey "read-only" block, separate from the editable fields. For each object, `source` says who owns it:

| source | meaning | box colour in UI |
|---|---|---|
| `auto` | detector output, untouched | blue |
| `edited` | detector output that a human changed; the detector's original answer stays in `auto_values` | amber |
| `manual` | added by a human | amber |
| `composed` | placed by the composer (exact by construction) | teal |

### "Manual corrections are the source of truth"

* Detection runs **once**, on upload. Nothing re-runs it implicitly (reads, edits, restarts).
* Re-running is an explicit action with two modes:
  * **Keep my edits** - replaces only untouched `auto` objects. Edited/manual objects stay, and objects the user *deleted* are remembered (tombstones) so they are not resurrected. New detections that overlap (IoU >= 0.5) a human-owned or deleted box are skipped.
  * **Replace everything** - back to pure detector output (the UI asks for confirmation if you have edits).
* If the detector **crashes**, the image is marked `failed` with the error and all existing objects are left exactly as they were.
* Edits are validated (box inside image, 0 <= confidence <= 1, label required, provenance fields not editable).

---

## 2. How detection works and why

Pipeline in `catalog/detector.py` (classical computer vision, no model download):

1. **Foreground.** Alpha channel if present -> distance from the uniform border colour in Lab space -> a **quadratic-surface background model** for smooth gradients (walls, vignettes) -> "flat artwork" mode when the whole frame is a palette (full-bleed colour swatches) -> GrabCut as a last, low-confidence resort.
2. **Colour regions.** Every foreground pixel gets a colour name; connected components per colour form regions.
3. **Grouping** (touching regions):
   * *person*: a skin-coloured head near the top of a tall figure -> **one** object (boy / girl, with clothing and hair colours);
   * *dominant region + small parts* (apple + stem + leaf) -> **one** object;
   * *comparable regions* (swatches, overlapping shapes) -> **one object per region**.
4. **Labelling.** Shape analysis (circle / ellipse / square / rectangle / triangle) plus a small lexicon (round red + leaf -> apple, round orange + calyx -> orange). **Occluded shapes** are classified from the *visible* part of their outline only (circle fit on the visible arc, or dominant straight-edge directions), so a circle half-hidden by a square is still a circle.
5. **Confidence** = f(shape-fit quality, contrast with background, size, occlusion). It is a **heuristic score, not a calibrated probability.** It is used for ranking and to flag low-confidence objects (< 0.6 -> dotted box + "needs review" in the UI).

Works for cartoons because cartoon art is flat colour with clean edges - exactly what colour-region segmentation likes. Works for photographs *on clean/gradient backgrounds* (product shots).

### Technology choices and trade-offs

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Classical CV (used)** | Offline, instant, deterministic, explainable, no weights/GPU, easy to test | Only a small vocabulary; fails on busy photos | Right baseline for this test set; ships and runs anywhere |
| YOLO / RT-DETR (ONNX) | Fast, real photo detection, 80+ classes | COCO classes only (no "circle", "cartoon boy"); weaker on cartoons; needs weights | Good second backend for photos |
| Open-vocabulary (Grounding-DINO / OWL-ViT) + SAM | Any label, good boxes and masks | GPU, heavier, slower | Best self-hosted production path |
| Hosted vision LLM | Best semantics and cartoon understanding, almost no code | Latency, cost, privacy, imprecise boxes, non-deterministic | **Used for photographs**, paired with GrabCut to tighten boxes |

**Design:** the classical stage stays the fast, exact, free layer for flat images; a hosted vision model handles photographs (implemented, section 2b). For a self-hosted production setup, replace the hosted call with an open-vocabulary detector + SAM behind the same interface: `CatalogService(detector=...)` accepts any object with `name` and `detect(PIL.Image) -> DetectionResult`; nothing else changes.

### 2b. Photographs: hybrid detection with a hosted vision model

The assignment requires detection to work for *non-cartoonized (photographic)* images and allows hosted AI services, so the system is hybrid:

```
image ──> ClassicalDetector (always runs first, ~30 ms)
            ├─ flat / cartoon image ............ keep the offline result (exact boxes, no network, no cost)
            └─ looks photographic, or no clean background found
                    └─> VisionDetector
                          1. send the image (JPEG, <=1280 px) to the model; ask for JSON: label, colour, box_2d, confidence
                          2. tighten each approximate box with GrabCut (models know *what*, edges are only approximate)
                          3. colour: the model's colour word if it is in our vocabulary, else measured from the object's pixels
```

* **Routing signal:** flat artwork is made of a few exact colours; a photograph is not. `is_photographic()` quantises the image and checks how much the 16 most common colours cover (< 80% -> photograph). Images with no clean background (offline GrabCut fallback) are also routed.
* **Measured effect** (synthetic photo with shading, grain and shadow; real photos will vary): offline detector alone -> one generic `object`, brown, box far too large; with the vision model -> `apple`, red, IoU 0.88 against the true box (0.67 before GrabCut tightening).
* **Failure behaviour:** if the model service is down and the offline detector found something, you get the offline result with a note saying so; if there is nothing to fall back on, the image is marked `failed` with the reason (e.g. `HTTP 401`), and any existing manual objects are untouched.
* **Manual corrections** work identically for model-detected objects (`source = auto` until edited).
* **Privacy / cost:** the image leaves your machine only if a key is configured and the image looks photographic (or `DETECTOR=vision`). Cartoons are never sent.
* **Confidence for model detections** is the model's own self-reported estimate (capped at 0.95, x0.85 if GrabCut could not refine the box). It is not calibrated.

**Enable it (pick one provider):**

| | Gemini (has a free tier) | Claude |
|---|---|---|
| Get a key | https://aistudio.google.com/apikey | https://console.anthropic.com |
| Windows PowerShell | `$env:GEMINI_API_KEY="..."` | `$env:ANTHROPIC_API_KEY="..."` |
| Mac / Linux | `export GEMINI_API_KEY="..."` | `export ANTHROPIC_API_KEY="..."` |
| Model (optional override) | `GEMINI_MODEL` (default `gemini-3.5-flash`) | `ANTHROPIC_MODEL` (default `claude-sonnet-5-5`) |

Set the variable in the same terminal, then `python scripts/check_vision.py` (a 10-second check: prints the detections and writes `vision_check.png`) and `python app.py`. The header of the UI shows which detector is active. Other switches: `VISION_PROVIDER=gemini|anthropic`, `DETECTOR=hybrid|classical|vision`. Model names change over time; if you get a 404, set the `*_MODEL` variable to a current model from the provider's docs.

**What is and is not verified:** the routing, parsing, GrabCut tightening, colour logic, fallbacks and the exact HTTP request shapes (taken from the providers' documentation) are covered by automated tests with the network faked. Calls to the live services were **not** run during development (no network access in my build environment), so run `scripts/check_vision.py` with your key once before recording the demo.

### Known limitations (deliberately stated)

* **The offline detector alone is not a photo detector.** Without an API key, photographs fall to GrabCut and yield one generic `object` at confidence 0.40, or nothing (`no_objects`) - the UI explains why and invites the user to draw boxes. With a key, photographs go to the vision model (section 2b). Model output can still be wrong or miss objects, which is what manual correction is for.
* **Labels come from rules.** `boy` vs `girl` uses weak cartoon cues (dress flare, wide pigtails); confidence is capped at 0.78 (0.60 when cues disagree). A plain red circle with no stem/leaf is labelled `circle`, not `apple` (the demo shows a user fixing exactly this).
* **Shaded photos** (gradients inside one object) can split into several colour regions.
* Thresholds were tuned on the 8 specified images plus 4 edge cases I generated; a larger labelled set would be needed to tune them properly (see Evaluation).

---

## 3. Composite generation

`"Create an image containing one red apple and one orange"`

1. **Parse** (rule-based, deterministic): counts (`two`, `3`), colour and size adjectives (`red`, `large`), nouns (plural handled), and layout words (`next to`, `left of`, `above`, `in front of`, `in a grid`, `in a column`).
2. **Resolve to real assets.** A noun maps to a detected object (tight crop) or to a whole catalog image by name. If several match, an image *named* like the noun wins, then confidence; the alternatives are recorded in the composite's provenance.
3. **Never fake an asset.** If anything is missing - `banana`, or `blue apple` when only a red one exists - the composer raises `MissingAssetError` (HTTP 422) naming what is missing and what the catalog contains. **Nothing is generated, substituted or partially saved.**
4. **Cut out** the asset using the same foreground model as the detector (holes such as eyes and highlights are kept opaque).
5. **Layout**: row (default, bottom-aligned), column, grid or overlap with z-order; scale by `sqrt(area)` so a wide strip does not shrink a tall figure; `large`/`small` modifiers; global shrink-to-fit.
6. **Register** as a new catalog entry (`kind = composite`) with its own metadata: the instruction, layout, and per-placement provenance (source image, source box, placed box). Every placed object gets an exact box with `source = composed`; an image-level asset such as `shapes` contributes each of its objects (circle, square, ...) mapped into composite coordinates.

The parser is deliberately simple and testable. For free-form language, swap it for an LLM that outputs the same `Plan` (items, counts, layout); resolution, cut-out, layout and the "never fake" guarantee are unchanged.

Limitations: relation parsing handles one relation phrase per instruction; a cut-out of an object that overlaps a neighbour includes some neighbour pixels; no rotation or perspective.

---

## 4. Mapping to the assignment's scenarios

| # | Scenario | Where to see it |
|---|---|---|
| 1 | Detect names, colours, confidence, boxes | `scripts/demo.py` scenario 1; `tests/test_catalog.py::DetectorTests` |
| 2 | Manually add an object when detection misses | blank image -> draw a box (UI) / scenario 2 |
| 3 | Edit name, colour, confidence, box | busy-background apple: `circle` -> `apple` (scenario 3) |
| 4 | Corrections persist until detection is re-run | scenario 4; `test_manual_corrections_persist_until_explicit_rerun` |
| 5 | Composite from `apple` + `orange` | scenario 5; `test_apple_and_orange` (includes a pixel check) |
| 6 | Composite with boy, girl, coloured shapes | scenario 6; `test_boy_girl_shapes` |
| 7 | Overlapping objects, image with nothing detectable | `overlapping_shapes` (occluded circle), `blank`, `noise_scene` |

Evidence is generated by `python scripts/demo.py`: `demo_output/report.md`, `demo_output/annotated/*.png`, `demo_output/overview.png`, `demo_output/composites/*.png`.

---

## 5. Evaluation (how I would measure accuracy and performance)

* **Detection quality:** label a set of images (start with ~100 cartoon + ~100 clean photos). Match predicted to ground-truth boxes at IoU >= 0.5; report precision / recall / F1 per label and mAP@0.5; separately report colour-name accuracy and mean IoU of matched boxes. Track the `no_objects` rate on images that do contain objects (missed detections).
* **Confidence usefulness:** plot reliability curves; if the score is to drive auto-accept vs "needs review", calibrate it (isotonic regression) and choose the threshold from the precision/recall trade-off.
* **Human effort (the metric that matters for a correction workflow):** corrections per image and time to fix, from the `source` and `edited_fields` data this system already records. Every correction is also a labelled training example.
* **Composition:** instruction -> expected asset multiset (exact-match accuracy), and "must refuse" cases (missing/colour-mismatch) for a 100% refusal rate. Visual checks: no box outside canvas, no overlap unless requested.
* **Performance:** p50/p95 latency per image size (currently ~30 ms for these images at <= 640 px analysis size), throughput, memory. For production, detection moves to a background queue.

---

## 6. Suggested demo video (3-4 minutes)

1. **Load** the sample set; open `shapes` - four objects with labels, colours, confidence and boxes; point at the blue (AI) boxes and the read-only "written by the system" block.
2. **Overlap:** open `overlapping_shapes`; note the circle is found although half hidden.
3. **Failed detection:** open `blank` -> banner; *Draw a box*, name it `pencil`, save (amber = yours).
4. **Correct a mistake:** upload `sample_images/busy_background_apple.png`; the detector says `circle`; select it, rename to `apple`, adjust the box (type or drag). Show the "Detector said: ..." audit line.
5. **Persistence:** with edits on `shapes` (rename a shape, delete another), press **Re-run detection - Keep my edits**: your work survives, the deleted object stays deleted. Then **Replace everything** to show the reset.
6. **Compose:** `one red apple and one orange`, then `a boy, a girl, and colored shapes`; open the new entries and show their composed objects and provenance.
7. **Refusal:** `a blue apple and a banana` -> clear error, nothing created.
8. **Photograph (needs your API key):** set the key, restart, upload `sample_images/photo_like_apple.png` or a real photo; show the header saying which model is used and the boxes/labels it returns. For contrast, show the same image without a key (generic `object`, note explaining why).
9. Close with the architecture diagram and the limitations section.
