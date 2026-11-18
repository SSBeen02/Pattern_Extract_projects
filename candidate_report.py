"""Export local ZNCC peaks for the fabric images without changing selection."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


SOURCE = Path(__file__).parent / "aligned_1024ver"
OUTPUT = Path(__file__).parent / "candidate_report.csv"
# Approximate targets explicitly identified in the conversation.
TARGETS = {
    ("pattern04", "y"): 240,
    ("pattern07", "y"): 268,
    ("pattern13", "x"): 86,
    ("pattern16", "y"): 102,
}


def rows_for_axis(image: Image.Image, image_name: str, axis: int):
    original_length = image.size[0 if axis == 1 else 1]
    scale = min(1.0, 1024 / max(image.size))
    size = (round(image.width * scale), round(image.height * scale))
    gray = image.convert("L")
    if size != image.size:
        gray = gray.resize(size, Image.Resampling.LANCZOS)
    gray = gray.filter(ImageFilter.GaussianBlur(max(size) / 256))
    array = np.asarray(gray, dtype=np.float32) / 255
    length = array.shape[axis]
    lo = max(1, math.ceil(length / 100))
    hi = min(length - 1, math.floor(length / 3))
    if hi <= lo or float(np.var(array)) < 0.0001:
        return []

    scores = np.full(hi + 1, np.nan, dtype=np.float32)
    for shift in range(lo, hi + 1):
        a, b = ((array[:, shift:], array[:, :-shift]) if axis == 1
                else (array[shift:, :], array[:-shift, :]))
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt(np.mean(a * a) * np.mean(b * b)))
        if denom > 1e-6:
            scores[shift] = float(np.mean(a * b) / denom)
    finite = scores[lo:hi + 1]
    if not np.isfinite(finite).any():
        return []

    median = float(np.nanmedian(finite))
    best = float(np.nanmax(finite))
    absolute = 0.45
    median_based = 1 - (1 - median) * 0.58
    best_based = best - 0.01
    threshold = max(absolute, median_based, best_based)
    axis_name = "x" if axis == 1 else "y"
    target = TARGETS.get((image_name, axis_name))
    rejected_axis = median > 0.995 or best < max(absolute, median_based)

    peaks = [d for d in range(lo + 1, hi)
             if scores[d] >= scores[d - 1] and scores[d] >= scores[d + 1]]
    accepted = [d for d in peaks if scores[d] >= threshold] if not rejected_axis else []
    chosen = min(accepted) if accepted else (int(np.nanargmax(finite)) + lo if not rejected_axis else None)
    rows = []
    for shift in peaks:
        score = float(scores[shift])
        rows.append({
            "image": image_name, "axis": axis_name,
            "period_px": round(shift * original_length / length, 1),
            "zncc": round(score, 6), "median": round(median, 6),
            "best": round(best, 6), "absolute_threshold": absolute,
            "median_threshold": round(median_based, 6),
            "best_threshold": round(best_based, 6),
            "effective_threshold": round(threshold, 6),
            "margin_to_threshold": round(score - threshold, 6),
            "passes": bool(not rejected_axis and score >= threshold),
            "selected": shift == chosen,
            "user_target_px_approx": target if target is not None else "",
        })
    return rows


def main():
    rows = []
    for path in sorted(SOURCE.glob("pattern*.jpg")):
        with Image.open(path) as image:
            rows.extend(rows_for_axis(image, path.stem, 1))
            rows.extend(rows_for_axis(image, path.stem, 0))
    fields = ["image", "axis", "period_px", "zncc", "median", "best",
              "absolute_threshold", "median_threshold", "best_threshold",
              "effective_threshold", "margin_to_threshold", "passes",
              "selected", "user_target_px_approx"]
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{OUTPUT} ({len(rows)} local peaks)")
    for row in rows:
        if row["user_target_px_approx"] != "" and abs(row["period_px"] - row["user_target_px_approx"]) <= 3:
            print(row["image"], row["axis"], "target", row["period_px"],
                  "ZNCC", row["zncc"], "passes", row["passes"])


if __name__ == "__main__":
    main()
