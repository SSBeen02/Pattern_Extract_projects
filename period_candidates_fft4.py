'''비교 순서 (FFT3의 기존 처리와 분리한 이동 NCC 검증 실험):
1. FFT3의 RGB 채널별 FFT / peak 선택 / 축별 다수결을 그대로 재사용한다.
2. 얻은 dx, dy를 각각 한 번만 반올림하고 정수 픽셀 1, 2, 3배 이동량을 만든다.
3. 가로 s 이동: RGB[:, :-s]와 RGB[:, s:]를, 세로 s 이동:
   RGB[:-s, :]와 RGB[s:, :]를 비교한다. wrap/padding 없이 겹치는 영역만 쓴다.
   각 RGB 채널의 평균을 따로 제거한 뒤 모든 채널을 합친 NCC를 계산한다.
4. 축별 NCC가 min_axis_ncc 이상이고 축별 최고점과의 차이가 tolerance 이내인
   후보 중 가장 작은 배수를 최소 반복 단위로 표시한다. 기준 미달이면 미판정한다.
   이는 상대적/경험적 판정이며 실제 반복 단위의 보장을 의미하지 않는다.
5. 9개 가로/세로 조합은 (x_ncc + y_ncc) / 2 내림차순으로 순위를 매긴다.
   최소 반복 단위 표시는 별도 컬럼이며 NCC 순위를 강제로 바꾸지 않는다.
패치 후보는 중앙 crop의 크기/좌표로 CSV에 기록한다. 평균화, 타일링, VGG,
이미지/feature map 저장 및 별도 시각화 모듈 호출은 이 실험에서 실행하지 않는다.
FFT 관련 설정(alpha 등)은 FFT3의 현재 값을 따른다.'''
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from period_candidates_fft3 import axis_period_profiles, load_rgb
from period_output_naming import candidate_prefix


def shifted_ncc(rgb: np.ndarray, shift: int, axis: str) -> tuple[float, int]:
    """Return pooled channel-centered NCC and the overlapping pixel count."""
    if axis not in ("x", "y"):
        raise ValueError("axis must be x or y")
    length = rgb.shape[1] if axis == "x" else rgb.shape[0]
    if shift <= 0 or shift >= length:
        return float("nan"), 0
    if axis == "x":
        first, second = rgb[:, :-shift], rgb[:, shift:]
    else:
        first, second = rgb[:-shift, :], rgb[shift:, :]
    first, second = first.astype(np.float64), second.astype(np.float64)
    first -= first.mean(axis=(0, 1), keepdims=True)
    second -= second.mean(axis=(0, 1), keepdims=True)
    denominator = float(np.sqrt(np.sum(first * first) * np.sum(second * second)))
    overlap = first.shape[0] * first.shape[1]
    if denominator <= 1e-12:
        return float("nan"), overlap
    return float(np.clip(np.sum(first * second) / denominator, -1.0, 1.0)), overlap


def axis_candidates(rgb, period, axis, tolerance, min_axis_ncc):
    candidates = []
    base_shift = max(1, round(period))
    for multiplier in range(1, 4):
        shift = base_shift * multiplier
        score, overlap = shifted_ncc(rgb, shift, axis)
        candidates.append(dict(multiplier=multiplier, shift=shift, ncc=score,
                               overlap_pixels=overlap, eligible=False, selected=False))
    finite = [c["ncc"] for c in candidates if math.isfinite(c["ncc"])]
    best = max(finite) if finite else float("nan")
    for candidate in candidates:
        score = candidate["ncc"]
        candidate["eligible"] = (math.isfinite(score) and score >= min_axis_ncc
                                  and best - score <= tolerance + 1e-12)
    selected = next((c for c in candidates if c["eligible"]), None)
    if selected is not None:
        selected["selected"] = True
    return candidates


def rank_candidates(rgb, peaks_x, peaks_y, tolerance=0.02, min_axis_ncc=0.8):
    if not peaks_x or not peaks_y:
        return []
    height, width = rgb.shape[:2]
    dx, dy = peaks_x[0].period_px, peaks_y[0].period_px
    xs = axis_candidates(rgb, dx, "x", tolerance, min_axis_ncc)
    ys = axis_candidates(rgb, dy, "y", tolerance, min_axis_ncc)
    rows = []
    for y in ys:
        for x in xs:
            pw, ph = x["shift"], y["shift"]
            left, top = round((width - pw) / 2), round((height - ph) / 2)
            score = (x["ncc"] + y["ncc"]) / 2
            rows.append(dict(
                rank="", dx=dx, dy=dy, dx_mult=x["multiplier"], dy_mult=y["multiplier"],
                patch_width=pw, patch_height=ph, left=left, top=top,
                right=left + pw, bottom=top + ph,
                x_ncc=x["ncc"], y_ncc=y["ncc"], shift_ncc_score=score,
                x_overlap_pixels=x["overlap_pixels"], y_overlap_pixels=y["overlap_pixels"],
                x_eligible=x["eligible"], y_eligible=y["eligible"],
                minimal_repeat_candidate=x["selected"] and y["selected"],
                tolerance=tolerance, min_axis_ncc=min_axis_ncc,
                status="ok" if math.isfinite(score) else "invalid_shift_or_zero_energy",
            ))
    rows.sort(key=lambda row: (
        not math.isfinite(row["shift_ncc_score"]),
        -row["shift_ncc_score"] if math.isfinite(row["shift_ncc_score"]) else 0,
        row["patch_width"] * row["patch_height"], row["dy_mult"], row["dx_mult"],
    ))
    for rank, row in enumerate(rows, 1):
        if row["status"] == "ok":
            row["rank"] = rank
    return rows


def main():
    parser = argparse.ArgumentParser(description="Rank FFT3 period multiples using original-image shift NCC.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=Path("outputs/fft4"))
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Maximum NCC gap from each axis best for minimum-period selection (default: 0.02).")
    parser.add_argument("--min-axis-ncc", type=float, default=0.8,
                        help="Minimum axis NCC for minimum-period selection (default: 0.8).")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or not 0 <= args.tolerance <= 2:
        parser.error("--tolerance must be finite and between 0 and 2")
    if not math.isfinite(args.min_axis_ncc) or not -1 <= args.min_axis_ncc <= 1:
        parser.error("--min-axis-ncc must be finite and between -1 and 1")
    _, rgb = load_rgb(str(args.image))
    _, _, _, peaks_x, peaks_y = axis_period_profiles(rgb)
    rows = rank_candidates(rgb, peaks_x, peaks_y, args.tolerance, args.min_axis_ncc)
    if not rows:
        raise SystemExit("No candidates: at least one axis has no FFT peak. No CSV was written.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{candidate_prefix(args.image, rows[0])}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", *rows[0]])
        writer.writeheader()
        for row in rows:
            writer.writerow({"image": str(args.image.resolve()), **{
                key: ("" if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in row.items()
            }})
    print(f"CSV: {path}")
    selected = next((row for row in rows if row["minimal_repeat_candidate"]), None)
    if selected:
        print(f"Minimum repeat candidate: {selected['dx_mult']}x{selected['dy_mult']} "
              f"({selected['patch_width']}x{selected['patch_height']}px)")
    else:
        print("Minimum repeat undecided: at least one axis did not meet the NCC criteria.")


if __name__ == "__main__":
    main()
