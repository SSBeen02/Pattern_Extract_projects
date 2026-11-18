"""Estimate horizontal and vertical repeat spacing in a flat fabric image.

Usage: python find_repeat_size.py image.jpg [-o visualization.png]
Requires: numpy, Pillow
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass(frozen=True)
class RepeatResult:
    width: float
    height: float
    found_x: bool
    found_y: bool


def _axis_period(gray: np.ndarray, axis: int, max_size: int = 1024) -> float | None:
    """Find the smallest strong whole-image translation on one axis."""
    image_axis_length = gray.shape[axis]
    # Keep images up to 1024 pixels at native resolution; downsample larger ones.
    scale = min(1.0, max_size / max(gray.shape))
    if scale < 1:
        im = Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255)))
        im = im.resize((round(gray.shape[1] * scale), round(gray.shape[0] * scale)), Image.Resampling.LANCZOS)
    else:
        im = Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255)))
    # Fabric weave is high-frequency noise relative to the printed motif.
    im = im.filter(ImageFilter.GaussianBlur(max(im.size) / 256))
    gray = np.asarray(im, dtype=np.float32) / 255
    actual_scale = gray.shape[axis] / image_axis_length
    lo = max(1, math.ceil(gray.shape[axis] / 100))  # At most 100 repeats.
    hi = min(gray.shape[axis] - 1, math.floor(gray.shape[axis] / 3))  # At least 3 repeats.
    if hi <= lo:
        return None

    # ZNCC correlation: higher means a better match.
    scores = np.full(hi + 1, np.nan, dtype=np.float32)
    if float(np.var(gray)) < 0.0001:
        return None
    for shift in range(lo, hi + 1):
        a, b = (gray[:, shift:], gray[:, :-shift]) if axis == 1 else (gray[shift:, :], gray[:-shift, :])
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt(np.mean(a * a) * np.mean(b * b)))
        scores[shift] = float(np.mean(a * b) / denom) if denom > 1e-6 else np.nan

    finite = scores[lo:hi + 1]
    if not np.isfinite(finite).any():
        return None
    baseline = float(np.nanmedian(finite))
    maximum = float(np.nanmax(finite))
    # A repeat must be substantially better than arbitrary translations.
    if baseline > 0.995 or maximum < max(0.45, 1 - (1 - baseline) * 0.58):
        return None
    # A candidate must satisfy all three thresholds and be a local maximum.
    threshold = max(0.45, 1 - (1 - baseline) * 0.58, maximum - 0.01)
    candidates = []
    for shift in range(lo + 1, hi):
        if scores[shift] >= threshold and scores[shift] >= scores[shift - 1] and scores[shift] >= scores[shift + 1]:
            candidates.append(shift)
    if not candidates:
        candidates = [int(np.nanargmax(scores[lo:hi + 1]) + lo)]
    chosen = min(candidates)
    # Parabolic interpolation gives a fractional-pixel estimate.
    offset = 0.0
    if lo < chosen < hi:
        left, center, right = map(float, scores[chosen - 1:chosen + 2])
        curvature = left - 2 * center + right
        if curvature < -1e-8:
            offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return round((chosen + offset) / actual_scale, 1)


def find_repeat_size(image: Image.Image | str | Path, max_size: int = 1024) -> RepeatResult | None:
    """Return repeat spacing in pixels, or None when neither axis repeats."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    x = _axis_period(gray, 1, max_size)
    y = _axis_period(gray, 0, max_size)
    if x is None and y is None:
        return None
    return RepeatResult(x if x is not None else 100.0,
                        y if y is not None else 100.0,
                        x is not None, y is not None)


def save_visualization(image: Image.Image | str | Path, result: RepeatResult | None,
                       output: str | Path) -> None:
    """Draw one central repeat rectangle and label its dimensions."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    w, h = canvas.size
    if result is not None:
        left = round((w - result.width) / 2)
        top = round((h - result.height) / 2)
        draw.rectangle((left, top, round(left + result.width), round(top + result.height)),
                       outline=(255, 35, 35, 230), width=2)
        label = f"Width: {result.width:.1f}px   Height: {result.height:.1f}px"
        if not result.found_x or not result.found_y:
            label += "   (100.0px = default)"
    else:
        label = "Repeat not found"
    font = ImageFont.truetype("arial.ttf", max(18, w // 42)) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((10, 10, min(w - 10, box[2] + 30), box[3] + 30), fill=(0, 0, 0, 200))
    draw.text((20, 18), label, font=font, fill=(255, 255, 255, 255))
    canvas.save(output)


def save_tiled_image(image: Image.Image | str | Path, result: RepeatResult,
                     output: str | Path) -> None:
    """Crop the central repeat patch and tile it around the output center."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    source = image.convert("RGB")
    width = max(1, min(source.width, round(result.width)))
    height = max(1, min(source.height, round(result.height)))
    left = (source.width - width) // 2
    top = (source.height - height) // 2
    patch = source.crop((left, top, left + width, top + height))

    canvas = Image.new("RGB", (1024, 1024))
    start_x = (1024 - width) // 2
    start_y = (1024 - height) // 2
    while start_x > 0:
        start_x -= width
    while start_y > 0:
        start_y -= height
    for y in range(start_y, 1024, height):
        for x in range(start_x, 1024, width):
            canvas.paste(patch, (x, y))
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", type=Path,
        default=Path(r"D:\GitHub\Pattern_Extract_projects\aligned_1024ver"),
        help="입력 이미지 파일 또는 폴더",
    )
    parser.add_argument(
        "-o", "--output-dir", "--output_dir", type=Path,
        default=Path(r"D:\GitHub\Pattern_Extract_projects\output_1111"),
        help="결과 이미지 저장 폴더",
    )
    parser.add_argument("--max-size", type=int, default=1024,
                        help="주기 탐색에 사용할 이미지의 최대 변 길이 (기본값: 1024)")
    args = parser.parse_args()
    if args.max_size < 32:
        parser.error("--max-size는 32 이상이어야 합니다")
    if args.image.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        images = sorted(p for p in args.image.iterdir()
                        if p.is_file() and p.suffix.lower() in extensions)
    elif args.image.is_file():
        images = [args.image]
    else:
        parser.error(f"입력 파일 또는 폴더를 찾을 수 없습니다: {args.image}")
    if not images:
        parser.error(f"입력 폴더에 이미지가 없습니다: {args.image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        output = args.output_dir / f"{image_path.stem}_patch.png"
        result = find_repeat_size(image_path, args.max_size)
        save_visualization(image_path, result, output)
        if result is not None:
            tiled_output = args.output_dir / f"{image_path.stem}_tiled.png"
            save_tiled_image(image_path, result, tiled_output)
        value = "못 찾음" if result is None else f"가로 {result.width:.1f}px, 세로 {result.height:.1f}px"
        print(f"{image_path.name}: {value} → {output}")


if __name__ == "__main__":
    main()
