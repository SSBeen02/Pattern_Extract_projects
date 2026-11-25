"""Sobel 기울기 벡터로 이미지의 가로/세로 반복 크기를 추정한다."""

from __future__ import annotations
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SAVE_SCORE_CSV = True
SCORE_TOLERANCE = 0.15
IMAGE_SIZE = (1024, 1024)


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


def _refine_period(scores: np.ndarray, shift: int, lo: int, hi: int) -> float:
    """이웃한 세 점수로 정수 이동량을 보정한다."""
    # 기본값은 정수 이동량 그대로이며, 경계가 아닌 내부 최솟값만 보간한다.
    offset = 0.0
    if lo < shift < hi:
        # 선택 지점의 좌/중/우 점수를 취해 국소 곡선을 포물선으로 근사한다.
        left, center, right = map(float, scores[shift - 1:shift + 2])
        # 2차 차분은 근사 포물선의 곡률이며, 양수일 때 아래로 볼록한 최솟값이다.
        curvature = left - 2 * center + right
        if curvature > 1e-8:
            # 꼭짓점의 소수 픽셀 오프셋을 구하되 인접한 정수 사이를 넘지 않게 제한한다.
            offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return round(shift + offset, 1)


def _axis_period(gx: np.ndarray, gy: np.ndarray, axis: int) -> AxisResult | None:
    """한 축의 기울기 벡터 차이를 비교해 가장 작은 허용 주기를 선택한다."""
    # 이미지 안에 대략 3~100회 반복이 들어오는 이동량 범위만 조사한다.
    lo = max(2, math.ceil(gx.shape[axis] / 100))
    hi = min(gx.shape[axis] - 1, math.floor(gx.shape[axis] / 3))
    if hi < lo:
        # 유효한 탐색 범위가 없으면 해당 축은 추정할 수 없다.
        return None

    # 이동량별 점수를 저장한다. 인덱스는 곧 이동 픽셀 수이며 범위 밖은 사용하지 않는다.
    scores = np.full(hi + 2, np.nan, dtype=np.float64)
    for shift in range(lo - 1, hi + 2):
        # 같은 위치의 기울기 벡터 쌍을 이동 방향에 맞춰 두 배열로 정렬한다.
        if axis == 1:
            # X축 비교: 오른쪽으로 shift 픽셀 이동한 영역과 겹치는 원래 영역을 비교한다.
            dx = gx[:, shift:] - gx[:, :-shift]
            dy = gy[:, shift:] - gy[:, :-shift]
        else:
            # Y축 비교: 아래쪽으로 shift 픽셀 이동한 영역과 겹치는 원래 영역을 비교한다.
            dx = gx[shift:, :] - gx[:-shift, :]
            dy = gy[shift:, :] - gy[:-shift, :]
        # 각 대응 위치에서 (Gx, Gy) 벡터의 유클리드 거리를 구해 평균 점수로 기록한다.
        scores[shift] = float(np.mean(np.hypot(dx, dy)))

    # lo와 hi도 바깥 이웃과 비교해 탐색 범위 경계의 국소 최솟값을 확인한다.
    local_minima = [
        shift for shift in range(lo, hi + 1)
        if scores[shift] <= scores[shift - 1]
        and scores[shift] <= scores[shift + 1]
        and (scores[shift] < scores[shift - 1] or scores[shift] < scores[shift + 1])
    ]
    if not local_minima:
        return None

    finite_scores = scores[lo:hi + 1]
    minimum = float(np.min(finite_scores))
    threshold = minimum + SCORE_TOLERANCE
    # 허용 범위 안의 국소 최솟값 중 가장 짧은 주기를 선택한다.
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
    # 최종 주기만 소수 위치 보정하고, 점수 및 선택 위치는 실제 정수 이동량 기준으로 반환한다.
    return AxisResult(_refine_period(scores, chosen, lo, hi),
                      float(scores[chosen]), candidates, chosen)


def _prepare_gradients(source: Image.Image, gaussian_output: str | Path | None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """1024×1024 이미지를 흐리게 만든 뒤 Sobel 기울기를 한 번 계산한다."""
    if source.size != IMAGE_SIZE:
        raise ValueError(f"이미지 크기는 정확히 1024x1024이어야 합니다: {source.size}")
    blurred = source.convert("L").filter(ImageFilter.GaussianBlur(1024 / 170))
    if gaussian_output is not None:
        blurred.save(gaussian_output)
    gray = np.asarray(blurred, dtype=np.float32) / 255
    return (cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5),
            cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5))


def find_repeat_size(image: Image.Image | str | Path,
                     gaussian_output: str | Path | None = None
                     ) -> tuple[AxisResult | None, AxisResult | None] | None:
    """(가로 축 결과, 세로 축 결과)를 계산한다."""
    if isinstance(image, Image.Image):
        gx, gy = _prepare_gradients(image, gaussian_output)
    else:
        with Image.open(image) as source:
            gx, gy = _prepare_gradients(source, gaussian_output)
    if np.mean(np.hypot(gx, gy)) < 1e-6:
        return None
    x = _axis_period(gx, gy, 1)
    y = _axis_period(gx, gy, 0)
    if x is None and y is None:
        # 어느 축에서도 반복 주기를 찾지 못하면 전체 결과가 없다.
        return None
    # 한 축만 찾은 경우에도 찾은 축은 AxisResult, 못 찾은 축은 None으로 반환한다.
    return x, y


def _rgb_image(image: Image.Image | str | Path) -> Image.Image:
    """이미지 객체 또는 파일을 RGB 이미지로 읽는다."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as source:
        return source.convert("RGB")


def save_visualization(image: Image.Image | str | Path,
                       result: tuple[AxisResult | None, AxisResult | None] | None,
                       output: str | Path) -> None:
    """중앙의 추정 반복 패치와 주기/점수를 표시한 이미지를 저장한다."""
    canvas = _rgb_image(image)
    # 반투명 오버레이를 지원하는 그리기 객체를 만들고 캔버스 크기를 얻는다.
    draw = ImageDraw.Draw(canvas, "RGBA")
    w, h = canvas.size
    if result is not None:
        x, y = result
        width = x.period_px if x is not None else 100.0
        height = y.period_px if y is not None else 100.0
        left, top = round((w - width) / 2), round((h - height) / 2)
        draw.rectangle((left, top, round(left + width), round(top + height)),
                       outline=(255, 35, 35, 230), width=2)
        label = f"Width: {width:.1f}px   Height: {height:.1f}px"
        scores = f"X score: {x.score:.3f}" if x is not None else "X score: N/A"
        scores += f"   Y score: {y.score:.3f}" if y is not None else "   Y score: N/A"
        if x is None or y is None:
            label += "   (100.0px = default)"
    else:
        # 결과 자체가 없으면 반복 패치 표시 없이 미검출 문구만 남긴다.
        label = "Repeat not found"
    # Windows Arial이 있으면 캔버스 너비에 비례한 크기를 쓰고, 없으면 PIL 기본 글꼴을 사용한다.
    font = (ImageFont.truetype("arial.ttf", max(18, w // 42))
            if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default())
    # 결과가 없는 경우 두 번째 줄의 점수 텍스트를 비워 둔다.
    score_label = scores if result is not None else ""
    # 각 텍스트의 실제 바운딩 박스를 측정해 검은 배경 패널 크기를 계산한다.
    box = draw.textbbox((0, 0), label, font=font)
    score_box = draw.textbbox((0, 0), score_label, font=font)
    line_height = box[3] - box[1] + 8
    # 캔버스 안쪽에 텍스트를 읽기 쉬운 반투명 검정 패널로 표시한다.
    draw.rectangle((10, 10, min(w - 10, max(box[2], score_box[2]) + 30),
                    22 + line_height * (2 if score_label else 1)), fill=(0, 0, 0, 200))
    # 패널 첫 줄에 주기 설명을 쓰고, 점수가 있으면 바로 아래 줄에 점수도 추가한다.
    draw.text((20, 18), label, font=font, fill=(255, 255, 255, 255))
    if score_label:
        draw.text((20, 18 + line_height), score_label, font=font,
                  fill=(255, 255, 255, 255))
    # 완성된 RGB 캔버스를 지정 경로에 저장한다.
    canvas.save(output)


def save_tiled_image(image: Image.Image | str | Path,
                     result: tuple[AxisResult | None, AxisResult | None],
                     output: str | Path) -> None:
    """추정한 한 주기 패치를 1024×1024 캔버스 전체에 반복 배치해 저장한다."""
    source = _rgb_image(image)
    # 주기가 이미지보다 크거나 0에 가까워도 crop 크기가 유효하도록 범위를 제한한다.
    x, y = result
    width = max(1, min(source.width, round(x.period_px if x is not None else 100.0)))
    height = max(1, min(source.height, round(y.period_px if y is not None else 100.0)))
    # 선택한 크기의 패치를 이미지 중심에서 잘라낸다.
    left, top = (source.width - width) // 2, (source.height - height) // 2
    patch = source.crop((left, top, left + width, top + height))
    # 반복 결과를 담을 고정 크기의 검정 RGB 캔버스를 만든다.
    canvas = Image.new("RGB", (1024, 1024))
    # 패치가 캔버스 중앙에 오도록 초기 타일 위치를 잡는다.
    start_x, start_y = (1024 - width) // 2, (1024 - height) // 2
    # 중심에서 왼쪽/위쪽으로 역행하며 캔버스의 시작점을 덮는 위치까지 이동한다.
    while start_x > 0:
        start_x -= width
    while start_y > 0:
        start_y -= height
    # 캔버스가 모두 채워질 때까지 각 행과 열에 같은 패치를 붙인다.
    for y in range(start_y, 1024, height):
        for x in range(start_x, 1024, width):
            canvas.paste(patch, (x, y))
    # 반복 배치된 미리보기 이미지를 지정 경로에 쓴다.
    canvas.save(output)


def main() -> None:
    """명령행 인수를 검증하고 이미지별 분석, 시각화, CSV 출력을 수행한다."""
    # 모듈 설명을 도움말에 사용하고 실행 파일 기준 기본 입출력 경로를 구성한다.
    parser = argparse.ArgumentParser(description=__doc__)
    project_dir = Path(__file__).resolve().parent
    # 이미지 파일 하나 또는 폴더를 받을 선택 인수를 등록한다.
    parser.add_argument("image", nargs="?", type=Path,
                        default=project_dir / "aligned_1024ver",
                        help="입력 이미지 또는 이미지 폴더 (기본값: aligned_1024ver)")
    # 출력 폴더의 별칭과 기본 경로를 지정한다.
    parser.add_argument("-o", "--output-dir", "--output_dir", type=Path,
                        default=project_dir / "outputs_sobel",
                        help="결과 이미지 저장 폴더 (기본값: outputs_sobel)")
    args = parser.parse_args()
    if args.image.is_dir():
        # 폴더 입력이면 지원하는 확장자만 선택하고 이름 순으로 정렬해 처리 순서를 고정한다.
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        images = sorted(p for p in args.image.iterdir()
                        if p.is_file() and p.suffix.lower() in extensions)
    elif args.image.is_file():
        # 파일 입력이면 해당 파일만 분석 목록에 넣는다.
        images = [args.image]
    else:
        # 존재하지 않는 경로 또는 파일/폴더가 아닌 입력은 명령행 오류로 알린다.
        parser.error(f"입력 파일 또는 폴더를 찾을 수 없습니다: {args.image}")
    if not images:
        # 폴더에 지원 이미지가 없으면 출력 폴더를 만들기 전에 종료한다.
        parser.error(f"입력 폴더에 이미지가 없습니다: {args.image}")
    for image_path in images:
        with Image.open(image_path) as source:
            if source.size != IMAGE_SIZE:
                parser.error(
                    "이미지 크기는 정확히 1024x1024이어야 합니다: "
                    f"{image_path.name} ({source.width}x{source.height})"
                )

    # 기존 결과를 덮어쓰지 않도록 빈 번호의 폴더를 새로 만든다.
    # 우선 지정된 경로를 사용하고, 이미 있으면 뒤에 붙일 번호를 2부터 시작한다.
    output_dir = args.output_dir
    number = 2
    while True:
        try:
            # 부모 폴더가 없으면 함께 만들되, 대상 폴더가 이미 있으면 예외로 다음 번호를 시도한다.
            output_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            # 이미 사용 중인 이름은 보존하고, 번호가 붙은 다음 이름을 검사한다.
            output_dir = args.output_dir.with_name(f"{args.output_dir.name}{number}")
            number += 1
    # 모든 이미지의 후보 점수를 마지막에 모아 쓸 CSV 행 목록이다.
    rows = []
    for image_path in images:
        # 파일명에 patch 접미사를 붙인 시각화 출력 경로를 만든다.
        output = output_dir / f"{image_path.stem}_patch.png"
        # 반복 크기를 추정한 다음 원본 위 표시 이미지를 저장한다.
        gaussian_output = output_dir / f"{image_path.stem}_gaussian.png"
        result = find_repeat_size(image_path, gaussian_output)
        save_visualization(image_path, result, output)
        if result is not None:
            # 추정 결과가 존재할 때만 반복 타일 미리보기 파일도 생성한다.
            save_tiled_image(image_path, result, output_dir / f"{image_path.stem}_tiled.png")
        # 사람에게 출력할 요약 문자열을 검출 여부에 맞춰 구성하고 콘솔에 알린다.
        x, y = result if result is not None else (None, None)
        x_period = x.period_px if x is not None else 100.0
        y_period = y.period_px if y is not None else 100.0
        value = ("못 찾음" if result is None else
                 f"가로 {x_period:.1f}px, 세로 {y_period:.1f}px")
        print(f"{image_path.name}: {value} -> {output}, {gaussian_output}")
        # 축별 성공/실패를 CSV에 남기며, 성공한 축은 후보별 세부 정보를 기록한다.
        for axis, axis_result, selected_period in (
            ("x", x, x_period),
            ("y", y, y_period),
        ):
            if axis_result is None:
                rows.append({
                    "image": image_path.name, "axis": axis, "status": "failed",
                    "shift_px": "", "period_px": "", "score": "",
                    "pass": False, "selected": False, "selected_period_px": "",
                })
                continue
            for candidate in axis_result.candidates:
                rows.append({
                    "image": image_path.name, "axis": axis, "status": "success",
                    "shift_px": candidate.shift_px,
                    "period_px": candidate.period_px,
                    "score": f"{candidate.score:.3f}",
                    "pass": candidate.passed,
                    "selected": candidate.shift_px == axis_result.selected_shift,
                    "selected_period_px": (
                        selected_period
                        if candidate.shift_px == axis_result.selected_shift else ""
                    ),
                })
    if SAVE_SCORE_CSV:
        # UTF-8 BOM을 포함해 스프레드시트에서 한글이 깨지지 않도록 CSV를 작성한다.
        with (output_dir / "sobel_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            # 후보 정보의 열 이름과 순서를 지정하고, 헤더와 누적된 모든 후보 행을 쓴다.
            writer = csv.DictWriter(handle, fieldnames=("image", "axis", "status", "shift_px",
                                                         "period_px", "score", "pass", "selected",
                                                         "selected_period_px"))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    # 다른 모듈에서 import할 때는 실행하지 않고, 직접 실행할 때에만 CLI 처리를 시작한다.
    main()
