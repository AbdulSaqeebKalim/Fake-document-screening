"""
Run this right after `pip install -r requirements.txt` to confirm every
module works on your machine, with no server, no test images, and no
network required. It exercises each real algorithm against the official
ICAO 9303 sample MRZ and a synthetically generated clean-vs-tampered image
pair.

Usage:
    python scripts/smoke_test.py
"""
import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from app.services.mrz import parse_td3
from app.services import ela, risk_engine


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main():
    ok = True

    # --- MRZ: official ICAO 9303 sample vector ---
    l1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<"
    l2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    result = parse_td3(l1, l2)
    ok &= check("MRZ: official ICAO sample validates", result.valid)
    ok &= check("MRZ: surname parsed correctly", result.surname == "ERIKSSON")

    l2_bad = l2[:27] + ("3" if l2[27] != "3" else "4") + l2[28:]
    tampered = parse_td3(l1, l2_bad)
    ok &= check("MRZ: corrupted expiry check digit is caught", not tampered.valid)

    # --- ELA: clean vs. spliced synthetic image ---
    np.random.seed(0)
    base = (np.random.rand(300, 400, 3) * 50 + 150).astype("uint8")
    clean_img = Image.fromarray(base)
    buf = io.BytesIO(); clean_img.save(buf, "JPEG", quality=92); buf.seek(0)
    clean_img = Image.open(buf).convert("RGB")
    r_clean = ela.analyze(clean_img)

    tampered_arr = np.array(clean_img).copy()
    patch = (np.random.rand(90, 90, 3) * 255).astype("uint8")
    patch_img = Image.fromarray(patch)
    pbuf = io.BytesIO(); patch_img.save(pbuf, "JPEG", quality=35); pbuf.seek(0)
    patch_img = Image.open(pbuf).convert("RGB")
    tampered_arr[40:130, 60:150] = np.array(patch_img)
    tampered_img = Image.fromarray(tampered_arr)
    tbuf = io.BytesIO(); tampered_img.save(tbuf, "JPEG", quality=92); tbuf.seek(0)
    tampered_img = Image.open(tbuf).convert("RGB")
    r_tamp = ela.analyze(tampered_img)

    print(f"       ELA clean score:   {r_clean.tamper_score}")
    print(f"       ELA spliced score: {r_tamp.tamper_score}")
    ok &= check("ELA: spliced image scores higher than clean", r_tamp.tamper_score > r_clean.tamper_score)
    ok &= check("ELA: spliced image hotspot detected", r_tamp.hotspot_bbox is not None)

    # --- Risk engine: four demo-style scenarios land in the expected band ---
    scenarios = [
        ("clean", True, [], 5.0, False, 98.7, "low"),
        ("splice", True, [], 72.0, True, 41.2, "high"),
        ("forged_mrz", False, ["x"], 50.0, True, 95.4, "high"),
        ("borderline", True, [], 8.0, False, 76.3, "med"),
    ]
    for name, mrz_ok, fails, tamper_score, bbox, face, expected_band in scenarios:
        r = risk_engine.assess(mrz_ok, fails, tamper_score, bbox, face)
        ok &= check(f"Risk engine: '{name}' lands in '{expected_band}' band (got {r.band}, risk={r.risk})",
                    r.band == expected_band)

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED — see [FAIL] lines above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
