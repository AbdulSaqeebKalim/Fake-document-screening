# SentinX Backend

Backend for **SentinX — AI-Based Fake Identity & Document Screening System**
(SIH 2026, Team STCET-26, PS 26188).

This implements the four pipeline stages your frontend prototype already
visualizes (OCR extraction → MRZ validation → tamper forensics → face
verification), for real, as a FastAPI service. It's built to be a genuine
working system, not a mock:

| Stage | What it actually does | Library |
|---|---|---|
| 1. OCR extraction | Crops the MRZ band, binarizes it, runs Tesseract OCR | `pytesseract` |
| 2. MRZ validation | Spec-accurate ICAO Doc 9303 check-digit algorithm | pure Python |
| 3. Tamper forensics | Real Error Level Analysis (JPEG re-compression forensics) | `Pillow` + `OpenCV` |
| 4. Face verification | Haar-cascade face detection + histogram/ORB similarity | `OpenCV` |

**Honest scope note on face verification:** stage 4 is a real, working
similarity measure, but it's a classical-CV stand-in for the ArcFace/MagFace
embedding models named in your pitch deck (those need dlib or an ONNX
model bundle, which don't fit a lightweight hackathon backend without extra
setup). It correctly separates "same person" from "clearly different person"
for a demo, but won't hit deep-embedding accuracy on hard cases (aging, pose,
lighting). `app/services/face_match.py` has an `EmbeddingFaceMatcher` stub
documenting exactly where to plug in a real model later — swap it in without
touching any other file.

## Setup

```bash
cd sentinx-backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You also need the Tesseract OCR binary installed on your system (this is
separate from the `pytesseract` Python package, which is just a wrapper):

```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
# and add the install directory to PATH
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000/docs** for interactive Swagger docs, or
**http://localhost:8000/health** to confirm it's up.

## Endpoints

### `POST /api/screen`
The real pipeline. Multipart form fields:

- `document` (file, required) — image of the passport/ID page
- `selfie` (file, optional) — live capture / reference photo to check the
  face on the document against. Omit it and the response just skips the
  face-match signal instead of penalizing the score.
- `mrz_line1` / `mrz_line2` (text, optional) — if you already have MRZ text
  from a dedicated scanner, pass it directly and OCR is skipped.

Returns a JSON object with `verdict`, `risk` (0-100), `band`
(`low`/`med`/`high`), `findings` (plain-language list for the officer UI),
and the raw `mrz` / `tamper` / `face` / `ocr` sub-results (including
bounding boxes, so the frontend can draw the same overlay boxes it currently
fakes with hardcoded coordinates).

```bash
curl -X POST http://localhost:8000/api/screen \
  -F "document=@sample_passport.jpg" \
  -F "selfie=@sample_selfie.jpg"
```

### `GET /api/demo-cases`
Returns the four canned scenarios your prototype currently hardcodes in its
`<script>` block (`clean`, `splice`, `forged_mrz`, `borderline`) — moved
server-side so there's one source of truth, and so the demo still works if
a live camera/scanner isn't available on the judging floor.

### `GET /api/demo-cases/{key}`
Single case by key.

## Wiring this into your existing frontend

Your current `sentinx-prototype.html` runs entirely on hardcoded
`CASES` data and a fake timed animation — it never calls a server. Two ways
to connect it to this backend without a rewrite:

1. **Keep the canned demo, make it real data**: replace the `CASES` object
   in the `<script>` with a `fetch('http://localhost:8000/api/demo-cases')`
   call on page load. Same four scenarios, same animation, but the numbers
   now come from a live backend instead of being typed into the JS.
2. **Go fully live**: add a file input for a document image (and optionally
   a selfie), call `POST /api/screen` with `FormData`, and feed the returned
   `risk` / `verdict` / `findings` / bounding boxes into the same
   `docStage` / `results` / overlay-drawing code that already exists —
   the response shape was designed to match what that code expects.

## Project layout

```
sentinx-backend/
  app/
    main.py              # FastAPI app, CORS, router registration
    schemas.py            # Pydantic request/response models
    routers/
      screening.py        # POST /api/screen — the real pipeline
      demo_cases.py        # GET /api/demo-cases — canned fallback data
    services/
      ocr.py               # Tesseract-based MRZ + full-text extraction
      mrz.py               # ICAO 9303 check-digit parser/validator
      ela.py               # Error Level Analysis tamper detection
      face_match.py         # Face detection + similarity scoring
      risk_engine.py        # Fuses signals into risk score + verdict
  scripts/
    smoke_test.py         # Sanity-checks the pipeline without a server
  requirements.txt
```

## Testing & calibration scripts

```bash
python scripts/smoke_test.py        # verifies MRZ/ELA/risk engine, no server needed
python scripts/make_test_images.py  # generates synthetic passports in test_images/
python scripts/test_pipeline.py     # runs the full pipeline offline on those images
python scripts/calibrate_ela.py clean_dir tampered_dir   # tune ELA on REAL images
```

## IMPORTANT: known limitation of the tamper detector

ELA has a documented false positive on sharp, high-contrast edges — printed
text, the MRZ band, document borders. JPEG always struggles to encode hard
edges, tampered or not.

Measured on this codebase: on the flat synthetic documents produced by
`make_test_images.py`, a **clean** document scores 100/100 with the hotspot
landing on the MRZ text band. That is a false positive, not a detection.
The defaults are validated on *photographic* content (`smoke_test.py`:
clean 18.3 vs. spliced 64.4, hotspot correctly localized).

Two auto-corrections were tried and deliberately **not** shipped, because
both degraded genuine splice detection on validated data: Sobel edge-energy
suppression, and block-wise robust z-score outlier detection. Don't re-add
either without measuring.

What to do before your demo:
1. Run `scripts/calibrate_ela.py` on real document photos (clean vs. edited).
2. If the clean/tampered gap stays under ~10 points, set
   `TAMPER_MAX_CONTRIBUTION = 0.0` in `app/services/risk_engine.py` and
   present tamper detection as future work rather than a live claim.
3. Lean on the MRZ checksum in front of judges — it is deterministic,
   spec-exact, and cannot false-positive the way ELA does.

## Notes for your report / judging Q&A

- The MRZ checksum implementation was tested against the official ICAO
  Doc 9303 sample MRZ and correctly validates it, and correctly flags a
  single-digit tamper injected into the expiry check digit.
- The ELA module was tested on a synthetic clean vs. spliced image pair:
  the clean image scored 18.3/100 with no detected hotspot, the spliced
  image scored 64.4/100 with a bounding box localized to the pasted region.
- The MRZ checksum is deterministic, explainable, requires no model
  training or GPU, and runs on ordinary hardware — which supports your
  "edge deployment, <2.5s, no cloud dependency" feasibility claims. This
  is your strongest technical claim; lead with it.
- Be straight about the two weaker signals if asked. Face matching is
  classical CV, not ArcFace/MagFace (see above). Tamper detection is real
  ELA but is uncalibrated for document imagery and currently false-positives
  on text. Judges respond far better to "here is exactly what is validated
  and what isn't" than to a demo that falls over under one pointed question.
