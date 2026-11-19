"""Estimate fabric repeat spacing using Sobel gradient-vector distances.

Usage: python find_repeat_size_sobel.py image.jpg [-o output_dir]
Requires: numpy, Pillow, opencv-python
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# False로 바꾸면 결과 CSV를 저장하지 않는다.
SAVE_SCORE_CSV = True
# 축별 최소 점수에서 이 값 이내인 국소 최솟값을 통과시킨다.
SCORE_TOLERANCE = 0.15
# 큰 패치를 분할해 내부 반복을 다시 검사한다. False면 첫 번째 선택만 사용한다.
# # 내부 분할 점수의 최솟값에서 중앙값까지 허용할 비율.


@dataclass(frozen=True)
class AxisCandidate:
    shift_px: int
    period_px: float
    score: float
    passed: bool


@dataclass(frozen=True)
class AxisResult:
    period_px: float
    score: float
    candidates: tuple[AxisCandidate, ...]
    selected_shift: int


@dataclass(frozen=True)
class RepeatResult:
    width: float
    height: float
    found_x: bool
    found_y: bool
    score_x: float | None = None
    score_y: float | None = None
    candidates_x: tuple[AxisCandidate, ...] = ()
    candidates_y: tuple[AxisCandidate, ...] = ()
    selected_shift_x: int | None = None
    selected_shift_y: int | None = None


def _refine_period(scores: np.ndarray, shift: int, lo: int, hi: int,
                   actual_scale: float) -> float:
    offset = 0.0
    if lo < shift < hi:
        left, center, right = map(float, scores[shift - 1:shift + 2])
        curvature = left - 2 * center + right
        if curvature > 1e-8:
            offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return round((shift + offset) / actual_scale, 1)


def _axis_period(gray: np.ndarray, axis: int, max_size: int = 1024) -> AxisResult | None:
    """Choose the smallest near-optimal local minimum along one axis."""
    original_length = gray.shape[axis]
    scale = min(1.0, max_size / max(gray.shape))
    image = Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255)))
    if scale < 1:
        image = image.resize((round(gray.shape[1] * scale), round(gray.shape[0] * scale)),
                             Image.Resampling.LANCZOS)
    # 축별 분석 전에 원본 방식의 블러를 적용한다 (1024px 기준 반경 약 6px).
    image = image.filter(ImageFilter.GaussianBlur(max(image.size) / 170))
    gray = np.asarray(image, dtype=np.float32) / 255
    actual_scale = gray.shape[axis] / original_length

    # Search shifts corresponding to approximately 3–100 repeats.
    lo = max(1, math.ceil(gray.shape[axis] / 100))
    hi = min(gray.shape[axis] - 1, math.floor(gray.shape[axis] / 3))
    if hi < lo:
        return None

    # Differentiate the whole image once; cropping first creates edge artifacts.
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
    if np.mean(np.hypot(gx, gy)) < 1e-6:
        return None
    scores = np.full(hi + 1, np.nan, dtype=np.float64)
    for shift in range(lo, hi + 1):
        if axis == 1:
            dx = gx[:, shift:] - gx[:, :-shift]
            dy = gy[:, shift:] - gy[:, :-shift]
        else:
            dx = gx[shift:, :] - gx[:-shift, :]
            dy = gy[shift:, :] - gy[:-shift, :]
        # Average Euclidean distances between corresponding (Gx, Gy) vectors.
        scores[shift] = float(np.mean(np.hypot(dx, dy)))

    finite_scores = scores[lo:hi + 1]
    minimum = float(np.min(finite_scores))
    threshold = minimum + SCORE_TOLERANCE
    # 양쪽 이웃보다 작거나 같고 한쪽보다는 엄격히 작은 지점만 국소 최솟값이다.
    local_minima = [
        shift for shift in range(lo + 1, hi)
        if scores[shift] <= scores[shift - 1]
        and scores[shift] <= scores[shift + 1]
        and (scores[shift] < scores[shift - 1] or scores[shift] < scores[shift + 1])
    ]
    accepted = [shift for shift in local_minima if scores[shift] <= threshold]
    # 경계가 전역 최솟값이거나 평평한 곡선이면 전역 최솟값을 보존한다.
    if not accepted:
        accepted = [int(np.argmin(finite_scores) + lo)]
    chosen = min(accepted)
    # 점수는 선택된 정수 shift의 실제 측정값이다. 보간은 주기 위치에만 적용한다.
    ranked = sorted(local_minima or [int(np.argmin(finite_scores) + lo)],
                    key=lambda shift: (scores[shift], shift))[:4]
    candidates = tuple(
        AxisCandidate(shift_px=shift, period_px=round(shift / actual_scale, 1),
                      score=float(scores[shift]), passed=bool(scores[shift] <= threshold))
        for shift in ranked
    )
    return AxisResult(_refine_period(scores, chosen, lo, hi, actual_scale),
                      float(scores[chosen]), candidates, chosen)


def find_repeat_size(image: Image.Image | str | Path, max_size: int = 1024) -> RepeatResult | None:
    """Estimate horizontal and vertical periods in original-image pixels."""
    if isinstance(image, Image.Image):
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    else:
        with Image.open(image) as source:
            gray = np.asarray(source.convert("L"), dtype=np.float32) / 255
    x = _axis_period(gray, 1, max_size)
    y = _axis_period(gray, 0, max_size)
    if x is None and y is None:
        return None
    return RepeatResult(x.period_px if x is not None else 100.0,
                        y.period_px if y is not None else 100.0, x is not None, y is not None,
                        x.score if x is not None else None,
                        y.score if y is not None else None,
                        x.candidates if x is not None else (),
                        y.candidates if y is not None else (),
                        x.selected_shift if x is not None else None,
                        y.selected_shift if y is not None else None)


def save_visualization(image: Image.Image | str | Path, result: RepeatResult | None,
                       output: str | Path) -> None:
    if isinstance(image, Image.Image):
        canvas = image.convert("RGB")
    else:
        with Image.open(image) as source:
            canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    w, h = canvas.size
    if result is not None:
        left, top = round((w - result.width) / 2), round((h - result.height) / 2)
        draw.rectangle((left, top, round(left + result.width), round(top + result.height)),
                       outline=(255, 35, 35, 230), width=2)
        label = f"Width: {result.width:.1f}px   Height: {result.height:.1f}px"
        scores = (f"X score: {result.score_x:.3f}" if result.score_x is not None else "X score: N/A")
        scores += (f"   Y score: {result.score_y:.3f}"
                   if result.score_y is not None else "   Y score: N/A")
        if not result.found_x or not result.found_y:
            label += "   (100.0px = default)"
    else:
        label = "Repeat not found"
    font = (ImageFont.truetype("arial.ttf", max(18, w // 42))
            if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default())
    score_label = scores if result is not None else ""
    box = draw.textbbox((0, 0), label, font=font)
    score_box = draw.textbbox((0, 0), score_label, font=font)
    line_height = box[3] - box[1] + 8
    draw.rectangle((10, 10, min(w - 10, max(box[2], score_box[2]) + 30),
                    22 + line_height * (2 if score_label else 1)), fill=(0, 0, 0, 200))
    draw.text((20, 18), label, font=font, fill=(255, 255, 255, 255))
    if score_label:
        draw.text((20, 18 + line_height), score_label, font=font,
                  fill=(255, 255, 255, 255))
    canvas.save(output)


def save_tiled_image(image: Image.Image | str | Path, result: RepeatResult,
                     output: str | Path) -> None:
    if isinstance(image, Image.Image):
        source = image.convert("RGB")
    else:
        with Image.open(image) as opened:
            source = opened.convert("RGB")
    width = max(1, min(source.width, round(result.width)))
    height = max(1, min(source.height, round(result.height)))
    left, top = (source.width - width) // 2, (source.height - height) // 2
    patch = source.crop((left, top, left + width, top + height))
    canvas = Image.new("RGB", (1024, 1024))
    start_x, start_y = (1024 - width) // 2, (1024 - height) // 2
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
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("image", nargs="?", type=Path,
                        default=project_dir / "aligned_1024ver",
                        help="입력 이미지 또는 이미지 폴더 (기본값: aligned_1024ver)")
    parser.add_argument("-o", "--output-dir", "--output_dir", type=Path,
                        default=project_dir / "outputs_sobel",
                        help="결과 이미지 저장 폴더 (기본값: outputs_sobel)")
    parser.add_argument("--max-size", type=int, default=1024,
                        help="분석 이미지의 최대 변 길이 (기본값: 1024)")
    
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

    # 기존 결과를 덮어쓰지 않도록 빈 번호의 폴더를 새로 만든다.
    output_dir = args.output_dir
    number = 2
    while True:
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            output_dir = args.output_dir.with_name(f"{args.output_dir.name}{number}")
            number += 1
    rows = []
    for image_path in images:
        output = output_dir / f"{image_path.stem}_patch.png"
        result = find_repeat_size(image_path, args.max_size)
        save_visualization(image_path, result, output)
        if result is not None:
            save_tiled_image(image_path, result, output_dir / f"{image_path.stem}_tiled.png")
        value = ("못 찾음" if result is None else
                 f"가로 {result.width:.1f}px, 세로 {result.height:.1f}px")
        print(f"{image_path.name}: {value} -> {output}")
        if result is not None:
            for axis, candidates, selected_period, selected_shift in (
                ("x", result.candidates_x, result.width, result.selected_shift_x),
                ("y", result.candidates_y, result.height, result.selected_shift_y),
            ):
                for candidate in candidates:
                    rows.append({
                        "image": image_path.name, "axis": axis,
                        "shift_px": candidate.shift_px,
                        "period_px": candidate.period_px,
                        "score": f"{candidate.score:.3f}",
                        "pass": candidate.passed,
                        "selected": candidate.shift_px == selected_shift,
                        "selected_period_px": selected_period if candidate.shift_px == selected_shift else "",
                    })
    if SAVE_SCORE_CSV:
        with (output_dir / "sobel_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image", "axis", "shift_px",
                                                         "period_px", "score", "pass", "selected",
                                                         "selected_period_px"))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
