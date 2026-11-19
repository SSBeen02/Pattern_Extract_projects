"""Compare shifted fabric regions using color SSIM and ZNCC.

Usage:
    python compare_ssim_candidates.py
    python compare_ssim_candidates.py aligned_1024ver/pattern07.jpg --axis y --shifts 134 268

The default experiment covers the ambiguous candidates discussed in this project.
Requires only numpy and Pillow.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_CASES = (
    ("pattern04.jpg", "y", (134, 268)),
    ("pattern07.jpg", "x", (86, 172, 259)),
    ("pattern08.jpg", "y", (102, 204, 307)),
)


def box_mean(array: np.ndarray, width: int = 11) -> np.ndarray:
    """Local box mean using an integral image; output has the input size."""
    radius = width // 2
    # 경계에서도 같은 크기의 창을 사용하도록 반사 패딩을 적용한다.
    padded = np.pad(array, radius, mode="reflect")
    # 적분 영상으로 각 위치의 width × width 영역 합을 빠르게 계산한다.
    integral = np.pad(padded.cumsum(0, dtype=np.float64).cumsum(1), ((1, 0), (1, 0)))
    return ((integral[width:, width:] - integral[:-width, width:]
             - integral[width:, :-width] + integral[:-width, :-width])
            / (width * width))


def color_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean local SSIM over RGB channels, with pixel values in [0, 1]."""
    # [0, 1] 범위의 픽셀에 맞춘 SSIM 안정화 상수.
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    channel_scores = []
    for channel in range(3):
        x = a[:, :, channel].astype(np.float64)
        y = b[:, :, channel].astype(np.float64)
        mx, my = box_mean(x), box_mean(y)
        # E[X²] - E[X]²로 국소 분산을 구하고 반올림 오차로 인한 음수를 제거한다.
        vx = np.maximum(0, box_mean(x * x) - mx * mx)
        vy = np.maximum(0, box_mean(y * y) - my * my)
        covariance = box_mean(x * y) - mx * my
        # 국소 밝기, 대비, 구조의 유사도를 합친 SSIM 맵을 채널별로 평균낸다.
        ssim = ((2 * mx * my + c1) * (2 * covariance + c2)
                / ((mx * mx + my * my + c1) * (vx + vy + c2)))
        channel_scores.append(float(np.mean(ssim)))
    return float(np.mean(channel_scores))


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    """평균 밝기를 제거한 두 영역의 정규화 상호상관을 계산한다."""
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.mean(x * x) * np.mean(y * y))
    # 분산이 거의 없는 균일한 영역은 상관계수가 정의되지 않는다.
    return float(np.mean(x * y) / denominator) if denominator > 1e-12 else float("nan")


def compare(path: Path, axis: str, shifts: tuple[int, ...]) -> list[dict]:
    """각 이동 후보에 대해 겹치는 영역의 ZNCC와 SSIM을 비교한다."""
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    # 원본과 가우시안 블러 버전을 [0, 1]로 정규화한다.
    raw = np.asarray(rgb, dtype=np.float32) / 255
    blurred = np.asarray(rgb.filter(ImageFilter.GaussianBlur(4)), dtype=np.float32) / 255
    gray_blurred = np.asarray(rgb.convert("L").filter(ImageFilter.GaussianBlur(4)), dtype=np.float32) / 255
    results = []
    for shift in shifts:
        if shift < 1 or shift >= raw.shape[1 if axis == "x" else 0]:
            raise ValueError(f"Invalid {axis} shift {shift} for {path}")
        # x는 좌우, y는 상하에서 shift만큼 떨어진 동일 크기 영역을 맞춘다.
        slices = ((slice(None), slice(shift, None)), (slice(None), slice(None, -shift))) if axis == "x" else (
            (slice(shift, None), slice(None)), (slice(None, -shift), slice(None)))
        (a_rows, a_cols), (b_rows, b_cols) = slices
        a_raw, b_raw = raw[a_rows, a_cols], raw[b_rows, b_cols]
        a_blur, b_blur = blurred[a_rows, a_cols], blurred[b_rows, b_cols]
        a_gray, b_gray = gray_blurred[a_rows, a_cols], gray_blurred[b_rows, b_cols]
        # 회색조 블러 ZNCC와 RGB 원본/블러 SSIM을 각각 기록한다.
        results.append({
            "image": path.name, "axis": axis, "shift_px": shift,
            "zncc_blurred": round(zncc(a_gray, b_gray), 6),
            "color_ssim_raw": round(color_ssim(a_raw, b_raw), 6),
            "color_ssim_blurred": round(color_ssim(a_blur, b_blur), 6),
        })
    return results


def main() -> None:
    """명령행 입력 또는 기본 후보를 비교하고 결과를 CSV로 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--axis", choices=("x", "y"))
    parser.add_argument("--shifts", nargs="+", type=int)
    parser.add_argument("-o", "--output", type=Path, default=Path("ssim_candidate_comparison.csv"))
    args = parser.parse_args()
    if args.image is not None and (args.axis is None or args.shifts is None):
        parser.error("--axis and --shifts are required with an image")
    # 이미지가 지정되지 않으면 DEFAULT_CASES의 후보들을 일괄 실행한다.
    cases = ((args.image, args.axis, tuple(args.shifts)),) if args.image else tuple(
        (Path(__file__).parent / "aligned_1024ver" / name, axis, shifts)
        for name, axis, shifts in DEFAULT_CASES)
    rows = [row for path, axis, shifts in cases for row in compare(path, axis, shifts)]
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
