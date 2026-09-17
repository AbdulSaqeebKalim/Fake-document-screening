"""
Generates synthetic passport-style test images you can feed to the live
upload panel, so you can demo the pipeline without needing real ID scans
(which you should NOT use for a public demo anyway — see note below).

Produces, in ./test_images/ :
  passport_clean.jpg    - valid MRZ check digits, single-generation JPEG
  passport_forged.jpg   - MRZ expiry check digit corrupted (stage 2 fails)
  passport_spliced.jpg  - photo region re-pasted at a different JPEG
                          quality, which is what real photo substitution
                          does to the compression history (stage 3 fires)
  selfie_match.jpg      - the same synthetic face as on the clean passport
  selfie_different.jpg  - a different synthetic face

PRIVACY NOTE for your demo: do not upload real passports/Aadhaar/PAN of
yourself or teammates to a demo machine. These synthetic images are safer
and make the same point to judges.

Usage:
    python scripts/make_test_images.py
"""
import io
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.services.mrz import compute_check_digit  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_images")

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "C:\\Windows\\Fonts\\consola.ttf",
    "C:\\Windows\\Fonts\\cour.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def build_mrz(doc_no, dob, expiry, surname, given, nationality="NVA", sex="F",
              break_expiry_check=False):
    """Build a TD3 MRZ with correct check digits (or a deliberately wrong one)."""
    doc_field = doc_no.ljust(9, "<")[:9]
    doc_check = str(compute_check_digit(doc_field))
    dob_check = str(compute_check_digit(dob))
    exp_check = str(compute_check_digit(expiry))
    if break_expiry_check:
        exp_check = str((int(exp_check) + 4) % 10)

    line1 = ("P<" + nationality + surname + "<<" + given).ljust(44, "<")[:44]
    optional = "<" * 14                     # positions 29-42 (indices 28-41)
    optional_check = str(compute_check_digit(optional))  # index 42
    partial = (doc_field + doc_check + nationality + dob + dob_check + sex
               + expiry + exp_check + optional + optional_check)
    assert len(partial) == 43, len(partial)
    # ICAO composite covers positions 1-10, 14-20, 22-43 (indices 0:10, 13:20, 21:43)
    composite_data = partial[0:10] + partial[13:20] + partial[21:43]
    composite = str(compute_check_digit(composite_data))
    line2 = partial + composite
    assert len(line2) == 44, len(line2)
    return line1, line2


def draw_face(draw, x, y, w, h, seed):
    """Draw a crude but face-detectable synthetic portrait."""
    rng = np.random.RandomState(seed)
    skin = tuple(int(v) for v in (200 + rng.randint(-35, 25),
                                  170 + rng.randint(-35, 25),
                                  150 + rng.randint(-35, 25)))
    draw.rectangle([x, y, x + w, y + h], fill=(225, 228, 235))
    cx, cy = x + w // 2, y + int(h * 0.52)
    fw, fh = int(w * 0.62), int(h * 0.68)
    draw.ellipse([cx - fw // 2, cy - fh // 2, cx + fw // 2, cy + fh // 2], fill=skin)
    eye_y = cy - fh // 8
    eye_dx = fw // 5
    er = max(2, fw // 14)
    for sx in (-eye_dx, eye_dx):
        draw.ellipse([cx + sx - er, eye_y - er, cx + sx + er, eye_y + er], fill=(255, 255, 255))
        draw.ellipse([cx + sx - er // 2, eye_y - er // 2, cx + sx + er // 2, eye_y + er // 2],
                     fill=(40, 30, 25))
    draw.line([cx, eye_y + er, cx, cy + fh // 10], fill=tuple(max(0, c - 45) for c in skin), width=2)
    draw.arc([cx - fw // 5, cy + fh // 12, cx + fw // 5, cy + fh // 3], 15, 165,
             fill=(130, 70, 70), width=2)
    hair = tuple(int(v) for v in (60 + rng.randint(0, 60), 45 + rng.randint(0, 40), 40 + rng.randint(0, 35)))
    draw.chord([cx - fw // 2, cy - fh // 2, cx + fw // 2, cy + fh // 6], 180, 360, fill=hair)


def make_passport(path, surname, given, doc_no, dob_disp, exp_disp, dob_mrz, exp_mrz,
                  face_seed, break_expiry_check=False, splice_photo=False):
    W, H = 900, 580
    img = Image.new("RGB", (W, H), (234, 229, 216))
    d = ImageDraw.Draw(img)

    f_title = load_font(22)
    f_key = load_font(12)
    f_val = load_font(17)
    f_mrz = load_font(25)

    d.rectangle([0, 0, W, 70], fill=(30, 48, 84))
    d.text((24, 24), "FEDERATIVE STATE OF NARVANA   PASSPORT", font=f_title, fill=(240, 240, 245))

    px, py, pw, ph = 40, 100, 190, 235
    draw_face(d, px, py, pw, ph, face_seed)
    d.rectangle([px, py, px + pw, py + ph], outline=(120, 120, 130), width=2)

    fields = [
        ("SURNAME", surname), ("GIVEN NAMES", given), ("PASSPORT NO.", doc_no),
        ("NATIONALITY", "NARVANIAN"), ("DATE OF BIRTH", dob_disp), ("DATE OF EXPIRY", exp_disp),
    ]
    fx, fy = 270, 105
    for i, (k, v) in enumerate(fields):
        col, row = i % 2, i // 2
        cx = fx + col * 310
        cy = fy + row * 78
        d.text((cx, cy), k, font=f_key, fill=(110, 110, 120))
        d.text((cx, cy + 18), v, font=f_val, fill=(15, 15, 20))

    l1, l2 = build_mrz(doc_no, dob_mrz, exp_mrz, surname.replace(" ", "<"),
                       given.replace(" ", "<"), break_expiry_check=break_expiry_check)
    d.rectangle([0, H - 130, W, H], fill=(243, 240, 231))
    d.text((22, H - 110), l1, font=f_mrz, fill=(10, 10, 15))
    d.text((22, H - 62), l2, font=f_mrz, fill=(10, 10, 15))

    if splice_photo:
        # Re-encode just the photo region at a much lower quality and paste
        # it back — this is the compression-history mismatch that real photo
        # substitution leaves behind, and what ELA is designed to surface.
        region = img.crop((px, py, px + pw, py + ph))
        buf = io.BytesIO()
        region.save(buf, "JPEG", quality=28)
        buf.seek(0)
        img.paste(Image.open(buf).convert("RGB"), (px, py))

    img.save(path, "JPEG", quality=92)
    return l1, l2


def make_selfie(path, face_seed):
    W, H = 320, 380
    img = Image.new("RGB", (W, H), (210, 214, 222))
    d = ImageDraw.Draw(img)
    draw_face(d, 30, 40, 260, 300, face_seed)
    img.save(path, "JPEG", quality=92)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    l1, l2 = make_passport(
        os.path.join(OUT_DIR, "passport_clean.jpg"),
        "DEALVARES", "MARISA JOY", "NV3381027",
        "14 MAR 1994", "02 JUL 2029", "940314", "290702",
        face_seed=7,
    )
    print("passport_clean.jpg   MRZ:", l2)

    _, l2f = make_passport(
        os.path.join(OUT_DIR, "passport_forged.jpg"),
        "OYELARAN", "TOMISIN", "NV2217450",
        "05 AUG 1996", "02 JUL 2031", "960805", "310702",
        face_seed=11, break_expiry_check=True,
    )
    print("passport_forged.jpg  MRZ:", l2f, "(expiry check digit deliberately wrong)")

    _, l2s = make_passport(
        os.path.join(OUT_DIR, "passport_spliced.jpg"),
        "KOVAC", "ANDRIJA", "NV5502981",
        "22 NOV 1988", "11 JAN 2028", "881122", "280111",
        face_seed=23, splice_photo=True,
    )
    print("passport_spliced.jpg MRZ:", l2s, "(photo region re-compressed)")

    make_selfie(os.path.join(OUT_DIR, "selfie_match.jpg"), face_seed=7)
    make_selfie(os.path.join(OUT_DIR, "selfie_different.jpg"), face_seed=91)
    print("selfie_match.jpg / selfie_different.jpg written")

    print("\nAll test images written to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
