"""Sobel 기울기 벡터의 유사도를 이용해 이미지의 가로/세로 반복 크기를 추정한다.

처리 흐름:
1. 명령행 인수에서 이미지 파일/폴더와 분석 크기, 결과 폴더를 읽는다.
2. 각 입력 이미지를 그레이스케일로 바꾸고 X축과 Y축의 반복 주기를 각각 찾는다.
3. 분석 결과를 원본 이미지 위에 표시하고, 추정된 반복 크기의 패치를 타일 이미지로 만든다.
4. 축별 후보 주기와 점수, 선택 여부를 CSV 파일로 기록한다.

사용법: python find_repeat_size_sobel.py image.jpg [-o output_dir]
필요 패키지: numpy, Pillow, opencv-python
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

# True이면 축별 반복 주기 후보의 점수 및 선택 결과를 CSV로 저장한다.
SAVE_SCORE_CSV = True
# 축별 점수의 전역 최솟값보다 이 값만큼 높은 국소 최솟값까지 후보로 인정한다.
SCORE_TOLERANCE = 0.15


@dataclass(frozen=True)
class AxisCandidate:
    """한 축에서 조사한 반복 간격과 그 점수, 허용 여부를 보관한다."""

    shift_px: int  # 축소된 분석 이미지에서 비교한 정수 픽셀 이동량.
    period_px: float  # 원본 이미지 픽셀 단위로 환산한 후보 반복 주기.
    score: float  # 이 이동량만큼 떨어진 기울기 벡터 간 평균 거리 점수.
    passed: bool  # 후보 점수가 허용 임계값 이내인지 나타낸다.


@dataclass(frozen=True)
class AxisResult:
    """한 축의 최종 주기 추정값과 선택 근거를 묶어 반환한다."""

    period_px: float  # 포물선 보간을 적용한 주기 추정값(원본 픽셀 단위).
    score: float  # 선택된 정수 이동량에서 실제 계산한 거리 점수.
    candidates: tuple[AxisCandidate, ...]  # 점수가 좋은 국소 최솟값 최대 4개.
    selected_shift: int  # 최종 주기로 선택된 정수 이동량(분석 이미지 픽셀 단위).


@dataclass(frozen=True)
class RepeatResult:
    """이미지의 두 축 추정값을 저장하며, 찾지 못한 축은 기본값으로 표시한다."""

    width: float  # 가로 방향(X축)의 반복 주기(원본 픽셀 단위).
    height: float  # 세로 방향(Y축)의 반복 주기(원본 픽셀 단위).
    found_x: bool  # 가로 방향 주기를 실제 분석으로 찾았는지 여부.
    found_y: bool  # 세로 방향 주기를 실제 분석으로 찾았는지 여부.
    score_x: float | None = None  # 가로 방향의 선택 후보 점수; 미검출이면 None.
    score_y: float | None = None  # 세로 방향의 선택 후보 점수; 미검출이면 None.
    candidates_x: tuple[AxisCandidate, ...] = ()  # 가로 방향의 점수 상위 후보들.
    candidates_y: tuple[AxisCandidate, ...] = ()  # 세로 방향의 점수 상위 후보들.
    selected_shift_x: int | None = None  # 가로 방향에서 선택한 분석용 정수 이동량.
    selected_shift_y: int | None = None  # 세로 방향에서 선택한 분석용 정수 이동량.


def _refine_period(scores: np.ndarray, shift: int, lo: int, hi: int,
                   actual_scale: float) -> float:
    """이웃한 세 점수로 정수 이동량을 보정하고 원본 픽셀 주기로 변환한다."""
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
    # 분석 이미지 이동량과 보정 오프셋을 원본/분석 배율로 나누고 소수 첫째 자리로 반올림한다.
    return round((shift + offset) / actual_scale, 1)


def _axis_period(gray: np.ndarray, axis: int, max_size: int = 1024) -> AxisResult | None:
    """한 축의 기울기 벡터 차이를 비교해 가장 작은 허용 주기를 선택한다."""
    # 원본 축 길이를 기억해 두어, 분석 후 축소된 픽셀 수를 원본 크기로 환산한다.
    original_length = gray.shape[axis]
    # 긴 변을 max_size 이하로 축소하되 이미 작은 이미지는 확대하지 않는다.
    scale = min(1.0, max_size / max(gray.shape))
    # PIL 필터/리사이즈를 위해 정규화된 부동소수점 명암을 8비트 그레이스케일로 변환한다.
    image = Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255)))
    if scale < 1:
        # 가로세로 비율을 유지해 분석 픽셀 수를 제한하고, 고품질 필터로 축소한다.
        image = image.resize((round(gray.shape[1] * scale), round(gray.shape[0] * scale)),
                             Image.Resampling.LANCZOS)
    # 축별 분석 전에 원본 방식의 블러를 적용한다 (1024px 기준 반경 약 6px).
    image = image.filter(ImageFilter.GaussianBlur(max(image.size) / 170))
    # 블러 결과를 정규화.
    gray = np.asarray(image, dtype=np.float32) / 255
    actual_scale = gray.shape[axis] / original_length

    # 이미지 안에 대략 3~100회 반복이 들어오는 이동량 범위만 조사한다.
    lo = max(1, math.ceil(gray.shape[axis] / 100))
    hi = min(gray.shape[axis] - 1, math.floor(gray.shape[axis] / 3))
    if hi < lo:
        # 유효한 탐색 범위가 없으면 해당 축은 추정할 수 없다.
        return None

    # 비교 대상 전체에서 먼저 Sobel 미분을 계산한다. 먼저 잘라내면 경계 인공물이 생길 수 있다.
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
    # 이미지 전체 기울기가 거의 없으면 이동 간 차이를 이용할 정보가 부족하므로 중단한다.
    if np.mean(np.hypot(gx, gy)) < 1e-6:
        return None
    # 이동량별 점수를 저장한다. 인덱스는 곧 이동 픽셀 수이며 범위 밖은 사용하지 않는다.
    scores = np.full(hi + 1, np.nan, dtype=np.float64)
    for shift in range(lo, hi + 1):
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

    # 유효한 조사 구간의 전역 최소 점수와 후보 허용 기준을 구한다.
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
    # 전역 최솟값에 충분히 가까운 국소 최솟값만 통과 후보로 남긴다.
    accepted = [shift for shift in local_minima if scores[shift] <= threshold]
    # 경계가 전역 최솟값이거나 평평한 곡선이면 전역 최솟값을 보존한다.
    if not accepted:
        accepted = [int(np.argmin(finite_scores) + lo)]
    # 허용 후보 중 가장 작은 이동량을 선택해 가장 짧은 반복 주기를 우선한다.
    chosen = min(accepted)
    # 점수는 선택된 정수 shift의 실제 측정값이다. 보간은 주기 위치에만 적용한다.
    # CSV/진단용으로 국소 최솟값을 점수순으로 정렬해 상위 네 개만 보존한다.
    ranked = sorted(local_minima or [chosen],
                    key=lambda shift: (scores[shift], shift))[:4]
    # 각 후보의 주기를 원본 해상도로 환산하고, 동일 임계값 통과 여부를 기록한다.
    candidates = tuple(
        AxisCandidate(shift_px=shift, period_px=round(shift / actual_scale, 1),
                      score=float(scores[shift]), passed=bool(scores[shift] <= threshold))
        for shift in ranked
    )
    # 최종 주기만 소수 위치 보정하고, 점수 및 선택 위치는 실제 정수 이동량 기준으로 반환한다.
    return AxisResult(_refine_period(scores, chosen, lo, hi, actual_scale),
                      float(scores[chosen]), candidates, chosen)


def find_repeat_size(image: Image.Image | str | Path, max_size: int = 1024) -> RepeatResult | None:
    """입력 이미지를 회색조로 바꾸고 가로/세로 반복 주기를 각각 계산한다."""
    if isinstance(image, Image.Image):
        # 이미 열린 PIL 이미지면 회색조 변환 후 0~1 범위 배열로 바꾼다.
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    else:
        # 경로나 문자열이면 파일을 열고, with 블록이 끝날 때 파일 핸들을 닫는다.
        with Image.open(image) as source:
            gray = np.asarray(source.convert("L"), dtype=np.float32) / 255
    # axis=1은 가로 방향, axis=0은 세로 방향의 주기 탐색이다.
    x = _axis_period(gray, 1, max_size)
    y = _axis_period(gray, 0, max_size)
    if x is None and y is None:
        # 어느 방향에서도 반복을 찾지 못하면 결과 객체 대신 None을 반환한다.
        return None
    # 한 축만 찾은 경우에는 해당 축만 found 플래그를 세우고, 다른 축에는 표시용 기본값 100px를 쓴다.
    return RepeatResult(x.period_px if x is not None else 100.0,
                        y.period_px if y is not None else 100.0, x is not None, y is not None,
                        # 찾지 못한 축의 점수, 후보, 선택 이동량은 각각 None 또는 빈 튜플로 둔다.
                        x.score if x is not None else None,
                        y.score if y is not None else None,
                        x.candidates if x is not None else (),
                        y.candidates if y is not None else (),
                        x.selected_shift if x is not None else None,
                        y.selected_shift if y is not None else None)


def save_visualization(image: Image.Image | str | Path, result: RepeatResult | None,
                       output: str | Path) -> None:
    """중앙의 추정 반복 패치와 주기/점수를 표시한 이미지를 저장한다."""
    if isinstance(image, Image.Image):
        # PIL 이미지 입력은 원본을 변경하지 않도록 RGB 복사본으로 변환한다.
        canvas = image.convert("RGB")
    else:
        # 파일 경로 입력도 파일을 연 뒤 RGB 캔버스로 읽어 둔다.
        with Image.open(image) as source:
            canvas = source.convert("RGB")
    # 반투명 오버레이를 지원하는 그리기 객체를 만들고 캔버스 크기를 얻는다.
    draw = ImageDraw.Draw(canvas, "RGBA")
    w, h = canvas.size
    if result is not None:
        # 추정 폭/높이를 기준으로 이미지 중앙에 반복 패치 사각형을 배치한다.
        left, top = round((w - result.width) / 2), round((h - result.height) / 2)
        draw.rectangle((left, top, round(left + result.width), round(top + result.height)),
                       outline=(255, 35, 35, 230), width=2)
        # 첫 줄에 폭/높이를, 둘째 줄에 가능한 축별 점수를 표시한다.
        label = f"Width: {result.width:.1f}px   Height: {result.height:.1f}px"
        scores = (f"X score: {result.score_x:.3f}" if result.score_x is not None else "X score: N/A")
        scores += (f"   Y score: {result.score_y:.3f}"
                   if result.score_y is not None else "   Y score: N/A")
        if not result.found_x or not result.found_y:
            # 미검출 방향의 숫자는 실제 추정값이 아니라 기본값임을 텍스트로 알린다.
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


def save_tiled_image(image: Image.Image | str | Path, result: RepeatResult,
                     output: str | Path) -> None:
    """추정한 한 주기 패치를 1024×1024 캔버스 전체에 반복 배치해 저장한다."""
    if isinstance(image, Image.Image):
        # 이미 열린 이미지는 RGB로 정규화한다.
        source = image.convert("RGB")
    else:
        # 경로 입력 이미지를 RGB로 읽고, 파일 핸들은 즉시 닫는다.
        with Image.open(image) as opened:
            source = opened.convert("RGB")
    # 주기가 이미지보다 크거나 0에 가까워도 crop 크기가 유효하도록 범위를 제한한다.
    width = max(1, min(source.width, round(result.width)))
    height = max(1, min(source.height, round(result.height)))
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
    # 분석 전 축소할 이미지의 최대 변 길이를 설정한다.
    parser.add_argument("--max-size", type=int, default=1024,
                        help="분석 이미지의 최대 변 길이 (기본값: 1024)")
    
    # 명령행 입력을 해석하고 지원 범위를 벗어난 분석 크기는 즉시 오류 처리한다.
    args = parser.parse_args()
    if args.max_size < 32:
        parser.error("--max-size는 32 이상이어야 합니다")
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
        result = find_repeat_size(image_path, args.max_size)
        save_visualization(image_path, result, output)
        if result is not None:
            # 추정 결과가 존재할 때만 반복 타일 미리보기 파일도 생성한다.
            save_tiled_image(image_path, result, output_dir / f"{image_path.stem}_tiled.png")
        # 사람에게 출력할 요약 문자열을 검출 여부에 맞춰 구성하고 콘솔에 알린다.
        value = ("못 찾음" if result is None else
                 f"가로 {result.width:.1f}px, 세로 {result.height:.1f}px")
        print(f"{image_path.name}: {value} -> {output}")
        if result is not None:
            # X/Y 방향 후보를 같은 방식으로 순회하며 선택 후보와 최종 주기를 연결한다.
            for axis, candidates, selected_period, selected_shift in (
                ("x", result.candidates_x, result.width, result.selected_shift_x),
                ("y", result.candidates_y, result.height, result.selected_shift_y),
            ):
                for candidate in candidates:
                    # 각 후보의 분석값과 선택 상태를 CSV 필드 구조에 맞춰 행으로 축적한다.
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
        # UTF-8 BOM을 포함해 스프레드시트에서 한글이 깨지지 않도록 CSV를 작성한다.
        with (output_dir / "sobel_scores.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            # 후보 정보의 열 이름과 순서를 지정하고, 헤더와 누적된 모든 후보 행을 쓴다.
            writer = csv.DictWriter(handle, fieldnames=("image", "axis", "shift_px",
                                                         "period_px", "score", "pass", "selected",
                                                         "selected_period_px"))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    # 다른 모듈에서 import할 때는 실행하지 않고, 직접 실행할 때에만 CLI 처리를 시작한다.
    main()
