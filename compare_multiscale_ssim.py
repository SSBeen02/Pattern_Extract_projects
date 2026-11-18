"""Compare candidate periods with five-scale color MS-SSIM.

Usage: python compare_multiscale_ssim.py
The default cases are the ambiguous fabric periods in this project.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from compare_ssim_candidates import DEFAULT_CASES, box_mean


WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
OUTPUT = Path(__file__).parent / "multiscale_ssim_comparison.csv"


def components(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean luminance and contrast-structure terms across RGB channels."""
    luminance, contrast_structure = [], []
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    for channel in range(3):
        x = a[:, :, channel].astype(np.float64)
        y = b[:, :, channel].astype(np.float64)
        mx, my = box_mean(x), box_mean(y)
        vx = np.maximum(0, box_mean(x * x) - mx * mx)
        vy = np.maximum(0, box_mean(y * y) - my * my)
        covariance = box_mean(x * y) - mx * my
        luminance.append(float(np.mean((2 * mx * my + c1) / (mx * mx + my * my + c1))))
        contrast_structure.append(float(np.mean((2 * covariance + c2) / (vx + vy + c2))))
    return float(np.mean(luminance)), float(np.mean(contrast_structure))


def half_size(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    image = Image.fromarray(np.uint8(np.clip(array * 255, 0, 255)), "RGB")
    image = image.resize((max(1, width // 2), max(1, height // 2)), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255


def multiscale_score(a: np.ndarray, b: np.ndarray) -> tuple[float, list[float]]:
    cs_values = []
    final_luminance = 1.0
    for level in range(5):
        final_luminance, cs = components(a, b)
        cs_values.append(cs)
        if level < 4:
            a, b = half_size(a), half_size(b)
    score = max(final_luminance, 1e-8) ** WEIGHTS[-1]
    for cs, weight in zip(cs_values, WEIGHTS):
        score *= max(cs, 1e-8) ** weight
    return score, cs_values


def main() -> None:
    rows = []
    for name, axis, shifts in DEFAULT_CASES:
        path = Path(__file__).parent / "aligned_1024ver" / name
        with Image.open(path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
        for shift in shifts:
            if axis == "x":
                a, b = rgb[:, shift:], rgb[:, :-shift]
            else:
                a, b = rgb[shift:, :], rgb[:-shift, :]
            score, cs = multiscale_score(a, b)
            row = {"image": name, "axis": axis, "shift_px": shift,
                   "ms_ssim": round(score, 6)}
            row.update({f"scale_{2 ** level}_cs": round(value, 6)
                        for level, value in enumerate(cs)})
            rows.append(row)
            print(row, flush=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
