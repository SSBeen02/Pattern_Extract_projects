import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PATCH_RE = re.compile(r"^(?P<prefix>.+)_period_patch_dx(?P<dx>\d+)_dy(?P<dy>\d+)\.png$")


def safe_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    a_flat -= float(a_flat.mean())
    b_flat -= float(b_flat.mean())
    denom = math.sqrt(float(np.sum(a_flat * a_flat)) * float(np.sum(b_flat * b_flat)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(a_flat * b_flat) / denom)


def patch_ncc(patch: Image.Image, shift_x: int, shift_y: int) -> float:
    arr = np.asarray(patch.convert("RGB")).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    values: List[float] = []
    if 0 < shift_x < w:
        values.append(normalized_cross_correlation(arr[:, : w - shift_x, :], arr[:, shift_x:, :]))
    if 0 < shift_y < h:
        values.append(normalized_cross_correlation(arr[: h - shift_y, :, :], arr[shift_y:, :, :]))
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def read_base_period(csv_path: Path) -> Tuple[float, float]:
    base_dx = float("nan")
    base_dy = float("nan")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dx = abs(float(row["dx"]))
            dy = abs(float(row["dy"]))
            if dx > 1e-6 and dy < 1e-6:
                base_dx = dx
            if dy > 1e-6 and dx < 1e-6:
                base_dy = dy
    return base_dx, base_dy


def find_original(input_dir: Path, prefix: str) -> Path | None:
    stem = re.sub(r"^\d{2}_", "", prefix)
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.stem == stem:
            return path
    return None


def format_metric(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def draw_ncc_montage(
    prefix: str,
    original_path: Path | None,
    patches: List[Tuple[int, int, Image.Image, float]],
    base_dx: float,
    base_dy: float,
    out_path: Path,
) -> None:
    ranked = sorted(patches, key=lambda item: item[3] if np.isfinite(item[3]) else -np.inf, reverse=True)

    cell_w, cell_h = 420, 260
    label_h = 58
    margin = 22
    title_h = 46
    original_preview_h = 260
    section_title_h = 34
    grid_h = section_title_h + (cell_h + label_h) * 4
    canvas_w = margin * 2 + cell_w * 4
    canvas_h = margin * 3 + title_h + original_preview_h + grid_h
    montage = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(montage)
    title_font = safe_font(20)
    section_font = safe_font(18)
    label_font = safe_font(12)

    draw.text(
        (margin, margin),
        f"NCC patch ranking: {prefix}, dx={base_dx:.1f}px, dy={base_dy:.1f}px",
        fill=(0, 0, 0),
        font=title_font,
    )

    if original_path and original_path.exists():
        original_preview = Image.open(original_path).convert("RGB")
        original_preview.thumbnail((canvas_w - margin * 2, original_preview_h), Image.Resampling.LANCZOS)
        original_x = margin + (canvas_w - margin * 2 - original_preview.size[0]) // 2
        original_y = margin + title_h
        montage.paste(original_preview, (original_x, original_y))
        center_x = original_x + original_preview.size[0] // 2
        center_y = original_y + original_preview.size[1] // 2
        draw.line([(center_x, original_y), (center_x, original_y + original_preview.size[1])], fill=(255, 0, 0), width=2)
        draw.line([(original_x, center_y), (original_x + original_preview.size[0], center_y)], fill=(255, 0, 0), width=2)
        draw.rectangle(
            [original_x, original_y, original_x + original_preview.size[0], original_y + original_preview.size[1]],
            outline=(210, 210, 210),
            width=1,
        )
    draw.text((margin, margin + title_h + original_preview_h + 4), "Original image", fill=(0, 0, 0), font=label_font)

    grid_top = margin * 2 + title_h + original_preview_h
    draw.text((margin, grid_top), "NCC ranking (higher is better)", fill=(0, 0, 0), font=section_font)
    grid_top += section_title_h

    for rank, (dx_mult, dy_mult, patch, ncc) in enumerate(ranked, start=1):
        col = (rank - 1) % 4
        row = (rank - 1) // 4
        cell_left = margin + col * cell_w
        cell_top = grid_top + row * (cell_h + label_h)
        crop_area_w = 170
        tile_area_w = cell_w - crop_area_w - 18
        preview_h = cell_h - 28

        patch_copy = patch.copy()
        patch_copy.thumbnail((crop_area_w - 10, preview_h), Image.Resampling.LANCZOS)
        px = cell_left + (crop_area_w - patch_copy.size[0]) // 2
        py = cell_top + 22 + (preview_h - patch_copy.size[1]) // 2
        montage.paste(patch_copy, (px, py))

        tiled = Image.new("RGB", (patch.size[0] * 3, patch.size[1] * 3))
        for tile_y in range(3):
            for tile_x in range(3):
                tiled.paste(patch, (tile_x * patch.size[0], tile_y * patch.size[1]))
        tiled.thumbnail((tile_area_w - 10, preview_h), Image.Resampling.LANCZOS)
        tx = cell_left + crop_area_w + 14 + (tile_area_w - tiled.size[0]) // 2
        ty = cell_top + 22 + (preview_h - tiled.size[1]) // 2
        montage.paste(tiled, (tx, ty))

        draw.rectangle([cell_left, cell_top, cell_left + cell_w, cell_top + cell_h], outline=(210, 210, 210), width=1)
        draw.line([(cell_left + crop_area_w, cell_top), (cell_left + crop_area_w, cell_top + cell_h)], fill=(225, 225, 225), width=1)
        draw.text((cell_left + 8, cell_top + 5), "patch", fill=(80, 80, 80), font=label_font)
        draw.text((cell_left + crop_area_w + 14, cell_top + 5), "3x3 tiled", fill=(80, 80, 80), font=label_font)
        draw.text(
            (cell_left + 8, cell_top + cell_h + 5),
            f"#{rank} dx x{dx_mult}, dy x{dy_mult} ({patch.size[0]}x{patch.size[1]})",
            fill=(0, 0, 0),
            font=label_font,
        )
        draw.text((cell_left + 8, cell_top + cell_h + 25), f"NCC={format_metric(ncc)}", fill=(0, 0, 0), font=label_font)

    montage.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NCC-only patch montages from existing patch PNGs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    patch_dir = Path(args.patch_dir)
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List[Tuple[int, int, Path]]] = {}
    for path in patch_dir.glob("*_period_patch_dx*_dy*.png"):
        match = PATCH_RE.match(path.name)
        if not match:
            continue
        groups.setdefault(match.group("prefix"), []).append((int(match.group("dx")), int(match.group("dy")), path))

    made = 0
    for prefix, patch_paths in sorted(groups.items()):
        csv_path = csv_dir / f"{prefix}_fft_center_peak_candidates.csv"
        if not csv_path.exists():
            continue
        base_dx, base_dy = read_base_period(csv_path)
        if not np.isfinite(base_dx) or not np.isfinite(base_dy):
            continue
        shift_x = max(1, int(round(base_dx)))
        shift_y = max(1, int(round(base_dy)))
        patches: List[Tuple[int, int, Image.Image, float]] = []
        for dx_mult, dy_mult, path in sorted(patch_paths):
            patch = Image.open(path).convert("RGB")
            patches.append((dx_mult, dy_mult, patch, patch_ncc(patch, shift_x, shift_y)))
        if len(patches) != 16:
            continue
        original_path = find_original(input_dir, prefix)
        draw_ncc_montage(prefix, original_path, patches, base_dx, base_dy, output_dir / f"{prefix}_period_patches_ncc.png")
        made += 1

    print(f"montages_created={made}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
