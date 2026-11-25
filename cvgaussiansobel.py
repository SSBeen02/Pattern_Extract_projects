"""Sobel 기울기 벡터로 1024x1024 이미지의 가로/세로 반복 크기를 추정한다."""

from __future__ import annotations
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import cv2 as cv
import numpy as np

SAVE_SCORE_CSV = True
SCORE_TOLERANCE = 0.15
IMAGE_SIZE = 1024          # 입력 이미지는 반드시 1024x1024
BLUR_SIGMA = 2.0           # 가우시안 블러 시그마(px)
BLUR_KSIZE = 13            # 약 6*sigma + 1, 홀수
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class AxisCandidate:
    """한 축에서 조사한 이동량 후보와 점수."""

    shift_px: int
    period_px: float
    score: float
    passed: bool


@dataclass(frozen=True)
class AxisResult:
    """한 축의 최종 주기와 후보 정보."""

    period_px: float
    score: float
    candidates: tuple[AxisCandidate, ...]
    selected_shift: int


def read_image(path: str | Path) -> np.ndarray:
    """이미지를 BGR로 읽고 1024x1024인지 검사한다.

    cv.imread는 한글 등 비ASCII 경로에서 실패할 수 있어 imdecode로 읽는다.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv.imdecode(data, cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return check_size(image, Path(path).name)


def check_size(image: np.ndarray, name: str = "image") -> np.ndarray:
    """입력이 정확히 1024x1024가 아니면 오류를 낸다."""
    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        h, w = image.shape[:2]
        raise ValueError(f"이미지 크기는 {IMAGE_SIZE}x{IMAGE_SIZE}이어야 합니다: {name} ({w}x{h})")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """한글 경로에서도 저장되도록 imencode + tofile로 쓴다."""
    ok, buffer = cv.imencode(Path(path).suffix or ".png", image)
    if not ok:
        raise ValueError(f"이미지를 저장할 수 없습니다: {path}")
    buffer.tofile(str(path))


def _refine_period(scores: np.ndarray, shift: int, lo: int, hi: int) -> float:
    """이웃한 세 점수로 정수 이동량을 포물선 보간하고 소수 첫째 자리로 반올림한다."""
    offset = 0.0
    if lo < shift < hi:
        left, center, right = map(float, scores[shift - 1:shift + 2])
        # 2차 차분이 양수일 때만 아래로 볼록한 최솟값이다.
        curvature = left - 2 * center + right
        if curvature > 1e-8:
            offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return round(shift + offset, 1)


def _axis_period(gx: np.ndarray, gy: np.ndarray, axis: int) -> AxisResult | None:
    """한 축의 기울기 벡터 차이를 비교해 가장 작은 허용 주기를 선택한다."""
    length = gx.shape[axis]
    # 이미지 안에 대략 3~100회 반복이 들어오는 이동량 범위만 조사한다.
    lo = max(2, math.ceil(length / 100))
    hi = min(length - 1, math.floor(length / 3))
    if hi < lo:
        return None

    # 이동량별 점수를 저장한다. 인덱스는 곧 이동 픽셀 수이다.
    scores = np.full(hi + 2, np.nan, dtype=np.float64)
    for shift in range(lo - 1, hi + 2):
        if axis == 1:
            dx = gx[:, shift:] - gx[:, :-shift]
            dy = gy[:, shift:] - gy[:, :-shift]
        else:
            dx = gx[shift:, :] - gx[:-shift, :]
            dy = gy[shift:, :] - gy[:-shift, :]
        # (Gx, Gy) 차이 벡터의 유클리드 거리 평균
        scores[shift] = float(np.mean(cv.magnitude(dx, dy)))

    # lo와 hi도 바깥 이웃과 비교해 경계의 국소 최솟값을 확인한다.
    local_minima = [
        shift for shift in range(lo, hi + 1)
        if scores[shift] <= scores[shift - 1]
        and scores[shift] <= scores[shift + 1]
        and (scores[shift] < scores[shift - 1] or scores[shift] < scores[shift + 1])
    ]
    if not local_minima:
        return None

    finite_scores = scores[lo:hi + 1]
    threshold = float(np.min(finite_scores)) + SCORE_TOLERANCE
    # 허용 범위 안의 국소 최솟값 중 가장 짧은 주기를 선택해 배수(2P, 3P)를 피한다.
    accepted = [shift for shift in local_minima if scores[shift] <= threshold]
    chosen = min(accepted) if accepted else int(np.argmin(finite_scores) + lo)

    # 점수 상위 후보를 보존하되, 실제 선택된 이동량은 반드시 포함한다.
    ranked = sorted(
        (chosen, *sorted(
            (shift for shift in local_minima if shift != chosen),
            key=lambda shift: (scores[shift], shift),
        )[:3]),
        key=lambda shift: (scores[shift], shift),
    )
    candidates = tuple(
        AxisCandidate(shift_px=shift, period_px=float(shift),
                      score=float(scores[shift]), passed=bool(scores[shift] <= threshold))
        for shift in ranked
    )
    return AxisResult(_refine_period(scores, chosen, lo, hi),
                      float(scores[chosen]), candidates, chosen)


def find_repeat_size(image: np.ndarray | str | Path
                     ) -> tuple[AxisResult | None, AxisResult | None] | None:
    """회색조 변환 → 가우시안 블러 → Sobel 후 (가로 축 결과, 세로 축 결과)를 계산한다."""
    if not isinstance(image, np.ndarray):
        image = read_image(image)
    check_size(image)
    gray = image if image.ndim == 2 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255

    # OpenCV 가우시안 블러 (sigma = 2px). 커널은 sigma의 약 6배.
    gray = cv.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), BLUR_SIGMA)

    # 전체 이미지에서 먼저 Sobel 미분을 계산한다. 먼저 잘라내면 경계 인공물이 생긴다.
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
    # 기울기가 거의 없으면 비교할 정보가 없다.
    if np.mean(cv.magnitude(gx, gy)) < 1e-6:
        return None

    # axis=1은 가로 방향, axis=0은 세로 방향의 주기 탐색이다.
    x = _axis_period(gx, gy, 1)
    y = _axis_period(gx, gy, 0)
    if x is None and y is None:
        return None
    return x, y


def _patch_size(result: tuple[AxisResult | None, AxisResult | None]) -> tuple[int, int]:
    """추정 주기를 패치 크기로 바꾼다. 실패한 축은 기본값 100px, 범위는 1~1024로 제한한다."""
    x, y = result
    width = max(1, min(IMAGE_SIZE, round(x.period_px if x is not None else 100.0)))
    height = max(1, min(IMAGE_SIZE, round(y.period_px if y is not None else 100.0)))
    return width, height


def save_visualization(image: np.ndarray | str | Path,
                       result: tuple[AxisResult | None, AxisResult | None] | None,
                       output: str | Path) -> None:
    """중앙의 추정 반복 패치와 주기/점수를 표시한 이미지를 저장한다."""
    canvas = (read_image(image) if not isinstance(image, np.ndarray) else image).copy()
    h, w = canvas.shape[:2]
    if result is not None:
        x, y = result
        width = x.period_px if x is not None else 100.0
        height = y.period_px if y is not None else 100.0
        left, top = round((w - width) / 2), round((h - height) / 2)
        cv.rectangle(canvas, (left, top), (round(left + width), round(top + height)),
                     (35, 35, 255), 2)  # BGR 빨강
        label = f"Width: {width:.1f}px   Height: {height:.1f}px"
        scores = f"X score: {x.score:.3f}" if x is not None else "X score: N/A"
        scores += f"   Y score: {y.score:.3f}" if y is not None else "   Y score: N/A"
        if x is None or y is None:
            label += "   (100.0px = default)"
        lines = [label, scores]
    else:
        lines = ["Repeat not found"]

    # 텍스트 크기에 맞춘 반투명 검정 패널 위에 글자를 쓴다.
    font, scale, thick = cv.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1700), 2
    sizes = [cv.getTextSize(t, font, scale, thick)[0] for t in lines]
    line_h = max(s[1] for s in sizes) + 14
    panel_w = max(s[0] for s in sizes) + 30
    overlay = canvas.copy()
    cv.rectangle(overlay, (10, 10), (min(w - 10, 10 + panel_w), 20 + line_h * len(lines)),
                 (0, 0, 0), -1)
    canvas = cv.addWeighted(overlay, 0.8, canvas, 0.2, 0)
    for i, text in enumerate(lines):
        cv.putText(canvas, text, (20, 10 + line_h * (i + 1)), font, scale,
                   (255, 255, 255), thick, cv.LINE_AA)
    write_image(output, canvas)


def save_tiled_image(image: np.ndarray | str | Path,
                     result: tuple[AxisResult | None, AxisResult | None],
                     output: str | Path) -> None:
    """추정한 한 주기 패치를 1024x1024 캔버스 전체에 반복 배치해 저장한다."""
    source = read_image(image) if not isinstance(image, np.ndarray) else image
    width, height = _patch_size(result)
    # 이미지 중심에서 한 주기 패치를 잘라낸다.
    left, top = (IMAGE_SIZE - width) // 2, (IMAGE_SIZE - height) // 2
    patch = source[top:top + height, left:left + width]
    # 패치가 캔버스 중앙에 오도록 위치별 패치 인덱스를 계산해 반복 배치한다.
    xs = (np.arange(IMAGE_SIZE) - left) % width
    ys = (np.arange(IMAGE_SIZE) - top) % height
    write_image(output, patch[np.ix_(ys, xs)])


def main() -> None:
    """명령행 인수를 검증하고 이미지별 분석, 시각화, CSV 출력을 수행한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("image", nargs="?", type=Path,
                        default=project_dir / "aligned_1024ver",
                        help="입력 이미지 또는 이미지 폴더 (기본값: aligned_1024ver)")
    parser.add_argument("-o", "--output-dir", "--output_dir", type=Path,
                        default=project_dir / "outputs_sobel",
                        help="결과 이미지 저장 폴더 (기본값: outputs_sobel)")
    args = parser.parse_args()

    if args.image.is_dir():
        images = sorted(p for p in args.image.iterdir()
                        if p.is_file() and p.suffix.lower() in EXTENSIONS)
    elif args.image.is_file():
        images = [args.image]
    else:
        parser.error(f"입력 파일 또는 폴더를 찾을 수 없습니다: {args.image}")
    if not images:
        parser.error(f"입력 폴더에 이미지가 없습니다: {args.image}")
    # 처리 시작 전에 모든 이미지가 1024x1024인지 먼저 검증한다.
    for image_path in images:
        try:
            read_image(image_path)
        except ValueError as error:
            parser.error(str(error))

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
        image = read_image(image_path)
        output = output_dir / f"{image_path.stem}_patch.png"
        result = find_repeat_size(image)
        save_visualization(image, result, output)
        if result is not None:
            save_tiled_image(image, result, output_dir / f"{image_path.stem}_tiled.png")
        x, y = result if result is not None else (None, None)
        x_period = x.period_px if x is not None else 100.0
        y_period = y.period_px if y is not None else 100.0
        value = ("못 찾음" if result is None else
                 f"가로 {x_period:.1f}px, 세로 {y_period:.1f}px")
        print(f"{image_path.name}: {value} -> {output}")
        for axis, axis_result, selected_period in (("x", x, x_period), ("y", y, y_period)):
            if axis_result is None:
                rows.append({
                    "image": image_path.name, "axis": axis, "status": "failed",
                    "shift_px": "", "period_px": "", "score": "",
                    "pass": False, "selected": False, "selected_period_px": "",
                })
                continue
            for candidate in axis_result.candidates:
                is_selected = candidate.shift_px == axis_result.selected_shift
                rows.append({
                    "image": image_path.name, "axis": axis, "status": "success",
                    "shift_px": candidate.shift_px,
                    "period_px": candidate.period_px,
                    "score": f"{candidate.score:.3f}",
                    "pass": candidate.passed,
                    "selected": is_selected,
                    "selected_period_px": selected_period if is_selected else "",
                })
    if SAVE_SCORE_CSV:
        with (output_dir / "sobel_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image", "axis", "status", "shift_px",
                                                         "period_px", "score", "pass", "selected",
                                                         "selected_period_px"))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()