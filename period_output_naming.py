"""Consistent image-id/rank/multiplier filenames for FFT3–FFT6 outputs."""
from __future__ import annotations

from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont

def image_id(image_path) -> str:
    stem = Path(image_path).stem
    match = re.search(r"(\d+)$", stem)
    return match.group(1).zfill(2) if match else stem


def candidate_prefix(image_path, row) -> str:
    rank = row["rank"] if row["rank"] != "" else "NA"
    return (f"{image_id(image_path)}_rank{rank}_"
            f"x{row['dx_mult']}_y{row['dy_mult']}")


def save_compact_montage(image, result, output_path) -> None:
    """Render FFT6's three-column patch/tile/original montage for any score mode."""
    row = result.metadata
    tiled = result.tiled
    if tiled is None:
        from period_candidates_fft3 import tile_patch_to_image

        pw, ph = result.image_patch.size
        left, top = round((image.width - pw) / 2), round((image.height - ph) / 2)
        tiled = tile_patch_to_image(result.image_patch, image.size,
                                    (left, top, left + pw, top + ph))

    scores = []
    for key, label in (("ncc_score", "NCC"), ("dense_score", "dense"),
                       ("conv3_4_score", "conv3_4"),
                       ("conv4_4_score", "conv4_4"),
                       ("combined_score", "combined")):
        if key in row:
            scores.append(f"{label}={row[key]:.6f}")
    if not scores and "vgg19_score" in row:
        label = f"VGG19 {row.get('vgg_layer', 'conv4_4')}" if row.get("vgg_feature") == "conv" else "VGG19 dense"
        scores.append(f"{label}={row['vgg19_score']:.6f}")
    if row["status"] != "ok":
        scores.append(row["status"])

    canvas = Image.new("RGB", (1170, 510), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((18, 16),
              f"Rank {row['rank'] or 'N/A'} | dx x{row['dx_mult']}, dy x{row['dy_mult']} | "
              f"patch {result.image_patch.width}x{result.image_patch.height}",
              fill="black", font=font)
    draw.text((18, 38), " | ".join(scores), fill="black", font=font)
    for column, (label, source) in enumerate((
        ("Averaged patch", result.image_patch),
        ("Tiled image", tiled),
        ("Original image", image),
    )):
        left = 20 + 390 * column
        draw.text((left, 74), label, fill="black", font=font)
        preview = source.convert("RGB").copy()
        preview.thumbnail((370, 390), Image.Resampling.LANCZOS)
        canvas.paste(preview, (left + (370 - preview.width) // 2,
                               100 + (390 - preview.height) // 2))
    canvas.save(output_path)


def save_ranked_patches(image, results, output_dir, score_method, image_path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        row = result.metadata
        if row["score_method"] != score_method:
            raise ValueError("Patch score method does not match visualization method")
        prefix = candidate_prefix(image_path, row)
        if row["dx_mult"] == row["dy_mult"] == 1:
            result.image_patch.save(output_dir / f"{image_id(image_path)}_original_patch.png")
        save_compact_montage(image, result, output_dir / f"{prefix}.png")
