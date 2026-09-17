"""
Calibrate the ELA tamper detector against YOUR real document images.

Why you need this: ELA's default parameters are validated on photographic
content, but every camera/scanner has a different JPEG pipeline, and sharp
printed text produces false positives (see the big note at the top of
app/services/ela.py). Running this on real images tells you what settings
actually separate clean from tampered for your data — instead of trusting
defaults that were never tuned for it.

How to use:
  1. Make two folders of REAL document photos/scans:
         calib/clean/      - documents you know are untouched
         calib/tampered/   - the same documents after you edit them in an
                             image editor (change a date, paste a face) and
                             re-save as JPEG
     Even 3-5 images per folder is enough to see the trend.
  2. Run:  python scripts/calibrate_ela.py calib/clean calib/tampered
  3. Pick the (quality, scale) row with the largest, most consistent gap
     between the clean mean and the tampered mean, then set those as the
     defaults in ela.analyze().

If the gap never opens up meaningfully, that is a real result and worth
knowing: it means ELA is not discriminative on your document type, and you
should lean on the MRZ checksum signal instead rather than showing judges
a tamper score you can't stand behind.
"""
import glob
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.services import ela  # noqa: E402

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def load_images(folder):
    paths = []
    for e in EXTS:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(set(paths))


def score_folder(paths, quality, scale):
    scores, localized = [], 0
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        r = ela.analyze(img, quality=quality, scale=scale)
        scores.append(r.tamper_score)
        if r.hotspot_bbox is not None:
            localized += 1
    return scores, localized


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: python scripts/calibrate_ela.py <clean_dir> <tampered_dir>")
        return

    clean_dir, tamp_dir = sys.argv[1], sys.argv[2]
    clean = load_images(clean_dir)
    tamp = load_images(tamp_dir)

    if not clean or not tamp:
        print(f"Found {len(clean)} clean and {len(tamp)} tampered images — need at least one of each.")
        return

    print(f"Calibrating on {len(clean)} clean / {len(tamp)} tampered images\n")
    header = f"{'quality':>7} {'scale':>6} | {'clean avg':>9} {'tamp avg':>9} {'gap':>7} | {'clean loc':>9} {'tamp loc':>8}"
    print(header)
    print("-" * len(header))

    best = None
    for quality in (75, 85, 90, 95):
        for scale in (10, 15, 18, 25):
            cs, cl = score_folder(clean, quality, scale)
            ts, tl = score_folder(tamp, quality, scale)
            if not cs or not ts:
                continue
            c_avg = sum(cs) / len(cs)
            t_avg = sum(ts) / len(ts)
            gap = t_avg - c_avg
            print(f"{quality:>7} {scale:>6} | {c_avg:>9.1f} {t_avg:>9.1f} {gap:>+7.1f} | "
                  f"{cl:>4}/{len(cs):<4} {tl:>3}/{len(ts):<4}")
            # Prefer a big gap that does NOT come with clean images being
            # localized (a localized clean image is a false positive).
            penalty = (cl / len(cs)) * 25.0
            objective = gap - penalty
            if best is None or objective > best[0]:
                best = (objective, quality, scale, c_avg, t_avg, gap, cl, len(cs))

    if best:
        _, q, s, c_avg, t_avg, gap, cl, ncl = best
        print("\nBest separation:")
        print(f"  quality={q}, scale={s}  -> clean {c_avg:.1f} vs tampered {t_avg:.1f} (gap {gap:+.1f})")
        print(f"  clean images false-flagged with a hotspot: {cl}/{ncl}")
        if gap < 10:
            print("\n  WARNING: gap under 10 points is weak. ELA is probably not")
            print("  discriminative on these images — rely on the MRZ checksum signal")
            print("  and present the tamper score as advisory only.")
        if cl > 0:
            print("\n  WARNING: some clean documents were flagged with a hotspot.")
            print("  Expect false positives on text-heavy regions; consider raising")
            print("  the threshold in ela.analyze() before demoing.")
        print(f"\n  To apply: edit app/services/ela.py -> analyze(..., quality={q}, scale={s})")


if __name__ == "__main__":
    main()
