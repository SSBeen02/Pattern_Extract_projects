"""Compare repeat candidates using signed gradient-vector similarity.

Usage: python compare_gradient_candidates.py
Requires: numpy, Pillow
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from compare_ssim_candidates import DEFAULT_CASES


OUTPUT = Path(__file__).parent / "gradient_candidate_comparison.csv"


def gradient_similarity(image: np.ndarray, axis: str, shift: int) -> tuple[float, float]:
    """Return signed vector correlation and unsigned gradient-magnitude correlation."""
    gy, gx = np.gradient(image.astype(np.float32))
    if axis == "x":
        a_x, b_x = gx[:, shift:], gx[:, :-shift]
        a_y, b_y = gy[:, shift:], gy[:, :-shift]
    else:
        a_x, b_x = gx[shift:, :], gx[:-shift, :]
        a_y, b_y = gy[shift:, :], gy[:-shift, :]
    dot = float(np.mean(a_x * b_x + a_y * b_y))
    energy_a = float(np.mean(a_x * a_x + a_y * a_y))
    energy_b = float(np.mean(b_x * b_x + b_y * b_y))
    signed = dot / np.sqrt(energy_a * energy_b) if energy_a * energy_b > 1e-20 else float("nan")
    mag_a = np.sqrt(a_x * a_x + a_y * a_y)
    mag_b = np.sqrt(b_x * b_x + b_y * b_y)
    mag_a -= mag_a.mean()
    mag_b -= mag_b.mean()
    denominator = float(np.sqrt(np.mean(mag_a * mag_a) * np.mean(mag_b * mag_b)))
    magnitude = float(np.mean(mag_a * mag_b) / denominator) if denominator > 1e-20 else float("nan")
    return signed, magnitude


def main() -> None:
    rows = []
    for name, axis, shifts in DEFAULT_CASES:
        path = Path(__file__).parent / "aligned_1024ver" / name
        with Image.open(path) as source:
            gray = source.convert("L")
        for blur in (1, 4, 8):
            image = np.asarray(gray.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255
            for shift in shifts:
                signed, magnitude = gradient_similarity(image, axis, shift)
                rows.append({"image": name, "axis": axis, "shift_px": shift,
                             "blur_radius": blur,
                             "signed_gradient": round(signed, 6),
                             "gradient_magnitude": round(magnitude, 6)})
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
