import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from period_candidates_visualization import (
    draw_frequency_domain_peaks,
    draw_grid_patch_preview,
    draw_spatial_domain_peaks,
    save_axis_period_functions,
    save_patch_box_preview,
    save_patch_montage,
)


VISUAL_CANDIDATE_LIMIT = 2
FREQUENCY_SEARCH_BOX_SIZE = 256
FREQUENCY_DISPLAY_BOX_SIZE = 200
AXIS_PROFILE_MAX_K = 100
AXIS_DUPLICATE_K_THRESHOLD = 4.0
AXIS_PEAK_RELATIVE_THRESHOLD = 0.73
NCC_TOP_LOCAL_MAXIMA = 10


# Step 0: Candidate data saved to CSV and reused by the visualization steps.
@dataclass
class FftPeakCandidate:
    candidate_id: int
    peak_x: float
    peak_y: float
    kx: float
    ky: float
    dx: float
    dy: float
    period_px: float
    angle_deg: float
    candidate_score: float


@dataclass
class AxisPeriodPeak:
    axis: str
    k: float
    period_px: float
    amplitude: float
    source: str = ""


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rgb(path: str) -> Tuple[Image.Image, np.ndarray]:
    # Step 1: Load the input as RGB. The current pipeline does not convert to grayscale.
    image = Image.open(path).convert("RGB")
    return image, np.asarray(image).astype(np.float32)


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    """채널 또는 1D 신호를 평균 0, 표준편차 1에 가깝게 표준화한다.

    ``channel``은 RGB 이미지의 단일 채널 배열((height, width))일 수도 있고,
    이미지의 x/y 방향 평균 투영으로 만든 1D 신호((width) 또는 (height))일
    수도 있다. 입력의 shape과 샘플 순서는 유지하고 값의 기준과 스케일만
    바꾼다.

    평균을 빼면 전체 밝기나 기본 기준선이 제거된다. 이어서 표준편차로 나누면
    서로 다른 밝기와 대비를 가진 채널도 비슷한 스케일에서 FFT로 비교할 수
    있다. 이 과정은 픽셀 위치를 이동하거나 주기를 변경하지 않는다.
    """
    # Step 2a: 전체 평균을 빼서 신호의 중심을 0으로 이동한다.
    # 예를 들어 [100, 110, 120]은 평균 110을 빼고 [-10, 0, 10]이 된다.
    # 일정한 밝기 성분은 FFT에서 DC 성분(k=0)으로 모이므로, 반복 변화에
    # 집중하려면 FFT를 수행하기 전에 이 기준선을 제거해야 한다.
    channel = channel - float(channel.mean())

    # Step 2b: 평균 제거 후 값의 퍼짐 정도를 계산한다.
    # 표준편차가 크면 채널의 대비/변동이 크고, 작으면 거의 평평한 신호이다.
    std = float(channel.std())
    if std > 1e-6:
        # Step 2c: 충분히 변하는 신호만 표준편차로 나눈다.
        # 결과는 평균 0, 표준편차 약 1이 되어 채널별 절대 대비 차이를 줄인다.
        channel = channel / std
    else:
        # Step 2d: 상수에 가까운 신호는 표준편차가 사실상 0이다.
        # 이때 나누면 NaN/inf가 생기므로 나누지 않고 평균 제거 결과를 유지한다.
        pass

    # Step 2e: 후속 FFT 계산의 자료형과 메모리 사용량을 일관되게 유지한다.
    return channel.astype(np.float32)



def normalize_01(values: np.ndarray) -> np.ndarray:
    """배열 내부의 유한한 값들을 선형적으로 0~1 범위에 매핑한다.

    변환식은 ``(values - lo) / (hi - lo)``이다. 여기서 ``lo``와 ``hi``는
    입력 배열의 유한한 최솟값과 최댓값이다. 따라서 최솟값은 0, 최댓값은 1이
    되고 중간 값의 상대적인 순서는 유지된다. 입력의 shape은 그대로 보존된다.

    FFT 진폭처럼 절대 크기보다 배열 안에서 어느 값이 큰지가 중요한 데이터를
    시각화하거나 피크 비교용으로 바꿀 때 사용한다. NaN과 +/-inf는 범위를
    계산할 때 제외하지만, 그 위치를 자동으로 다른 값으로 치환하지는 않는다.
    """
    # Step 2f: 범위 계산에 사용할 유한한 값만 추린다.
    # np.isfinite()는 NaN, +inf, -inf가 아닌 값에서만 True를 반환한다.
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        # 유효한 값이 하나도 없으면 최소/최대값을 정할 수 없다.
        # 입력과 같은 shape의 0 배열을 반환해 후속 계산을 안전하게 종료한다.
        return np.zeros(values.shape, dtype=np.float32)

    # Step 2g: 유한한 값들의 양 끝 범위를 찾는다.
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        # hi == lo이면 모든 유효한 값이 같아 분모가 0이 된다.
        # 값 사이의 상대적인 차이도 없으므로 전체를 0으로 표현한다.
        return np.zeros(values.shape, dtype=np.float32)

    # Step 2h: lo~hi 구간을 0~1 구간으로 선형 변환한다.
    # 유한한 입력값은 0~1로 변환되며, 입력에 있던 NaN/inf는 자동 복구하지
    # 않으므로 해당 위치에는 비유한 값이 남을 수 있다.
    return ((values - lo) / (hi - lo)).astype(np.float32)


def fft_rgb_log_magnitude(rgb: np.ndarray) -> np.ndarray:
    # Step 3: Build the 2D frequency-domain background image for visualization only.
    # This 2D FFT image is not used to choose the final period candidates.
    h, w, _channels = rgb.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    magnitude_sq = np.zeros((h, w), dtype=np.float64)

    for channel_idx in range(3):
        prepared = normalize_channel(rgb[:, :, channel_idx]) * window
        spectrum = np.fft.fftshift(np.fft.fft2(prepared))
        magnitude_sq += np.abs(spectrum) ** 2

    return np.log1p(np.sqrt(magnitude_sq)).astype(np.float32)

#1d fft를 통해 x축,y축 peak 반환
def axis_period_profiles(
    rgb: np.ndarray,
    min_period: float,
    max_period: float,
    max_k: int = AXIS_PROFILE_MAX_K,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[AxisPeriodPeak], List[AxisPeriodPeak]]:  #이 함수가 이런 타입의 값 5개를 반환한다”는 타입 설명
    # Step 4: Collapse RGB image channels into x/y 1D signals and run 1D FFT.
    # 반복 무늬가 이미지의 어느 축 방향으로 나타나는지 축별로 분석한다.
    h, w, _channels = rgb.shape
    max_axis_k = min(w // 2, h // 2, max_k) #100
    ks = np.arange(1, max_axis_k + 1, dtype=np.float32)  #FFT 주파수 번호 배열
    amp_x_sq = np.zeros(max_axis_k, dtype=np.float64)
    amp_y_sq = np.zeros(max_axis_k, dtype=np.float64)
    window_x = np.hanning(w).astype(np.float32)
    window_y = np.hanning(h).astype(np.float32)

    for channel_idx in range(3):
        channel = rgb[:, :, channel_idx].astype(np.float32)
        # Step 4a: x uses column means; y uses row means.
        # 평균 투영으로 2D 텍스처를 각 축의 1D 주기 신호로 압축한다.
        signal_x = normalize_channel(channel.mean(axis=0)) * window_x
        signal_y = normalize_channel(channel.mean(axis=1)) * window_y
        # Hann window는 신호 양끝의 불연속을 줄여 FFT 누설을 완화한다.
        spectrum_x = np.fft.rfft(signal_x)
        spectrum_y = np.fft.rfft(signal_y)
        # DC 성분(bin 0)은 제거하고 관심 있는 k 범위의 에너지만 축적한다.
        amp_x_sq += np.abs(spectrum_x[1 : max_axis_k + 1]) ** 2
        amp_y_sq += np.abs(spectrum_y[1 : max_axis_k + 1]) ** 2

    amp_x = normalize_01(np.log1p(np.sqrt(amp_x_sq)).astype(np.float32)) #fft logscale 후 정규화
    amp_y = normalize_01(np.log1p(np.sqrt(amp_y_sq)).astype(np.float32))

    # Step 5: Apply the spatial-period limits as a valid k range.
    # 이미지 길이 N에서 k cycles가 나타나면 한 주기는 N/k 픽셀이다.
    low_freq_cut_x = int(math.ceil(w / max_period))
    low_freq_cut_y = int(math.ceil(h / max_period))
    high_freq_cut_x = int(math.floor(w / min_period))
    high_freq_cut_y = int(math.floor(h / min_period))

    valid_x = (ks >= low_freq_cut_x) & (ks <= high_freq_cut_x) #유효한 주파수 범위에 해당하는 FFT bin만 True로 표시
    valid_y = (ks >= low_freq_cut_y) & (ks <= high_freq_cut_y)
    if not np.any(valid_x):
        # 설정 범위가 FFT bin과 겹치지 않으면 전체 계산 범위를 fallback으로 사용한다.
        valid_x = np.ones(ks.shape, dtype=bool)
    if not np.any(valid_y):
        valid_y = np.ones(ks.shape, dtype=bool)

    # 가장 강한 기본 주기 하나씩을 선택해 후속 후보 생성에 전달한다.
    peaks_x = select_axis_amplitude_peaks("x", ks, amp_x, valid_x, w, count=1)
    peaks_y = select_axis_amplitude_peaks("y", ks, amp_y, valid_y, h, count=1)
    return ks, amp_x, amp_y, peaks_x, peaks_y




def subpixel_peak_1d(v_minus: float, v_zero: float, v_plus: float) -> float:
    # Step 6a: Estimate where the peak lies between integer k samples with a parabola.
    # 세 샘플을 꼭짓점으로 하는 포물선의 꼭짓점 위치를 계산한다.
    denom = v_minus - 2.0 * v_zero + v_plus
    if abs(denom) < 1e-9:
        # 곡률이 거의 없으면 보정할 근거가 없으므로 정수 위치를 사용한다.
        return 0.0
    offset = 0.5 * (v_minus - v_plus) / denom
    # 인접 샘플 사이를 벗어나지 않도록 보정량을 제한한다.
    return float(np.clip(offset, -0.5, 0.5))


def refine_profile_peak(ks: np.ndarray, values: np.ndarray, peak_index: int) -> Tuple[float, float]:
    # Step 6b: Refine the selected integer-k peak to one decimal place.
    # 경계에서는 좌우 샘플이 없으므로 포물선 보정을 적용할 수 없다.
    if peak_index <= 0 or peak_index >= values.size - 1:
        k = float(ks[peak_index])
        return round(k, 1), float(values[peak_index])
    offset = subpixel_peak_1d(
        float(values[peak_index - 1]),
        float(values[peak_index]),
        float(values[peak_index + 1]),
    )
    # FFT의 정수 bin보다 실제 피크가 약간 이동한 위치를 반영한다.
    k = round(float(ks[peak_index]) + offset, 1)
    value = float(values[peak_index])
    return k, value


def select_axis_amplitude_peaks(
    axis: str,
    ks: np.ndarray,
    amplitude: np.ndarray,
    valid: np.ndarray,
    image_size: int,
    count: int = 2,
) -> List[AxisPeriodPeak]:
    # Step 6: Pick the earliest local peak whose amplitude is close enough to the strongest peak.
    # This favors the first plausible fundamental period over later harmonics with slightly higher amplitude.
    peaks: List[AxisPeriodPeak] = []

    valid_indices = np.where(valid)[0]
    if valid_indices.size == 0:
        # 허용 주기 범위에 해당하는 FFT bin이 없으면 후보를 만들 수 없다.
        return peaks

    max_amplitude = float(np.max(amplitude[valid_indices]))
    threshold = AXIS_PEAK_RELATIVE_THRESHOLD
    local_peak_indices: List[int] = []
    for peak_index in valid_indices:
        peak_index = int(peak_index)
        # 현재 bin이 양옆보다 크고, 설정한 진폭 기준 이상인지 확인한다.
        left = float(amplitude[peak_index - 1]) if peak_index > 0 else -np.inf
        center = float(amplitude[peak_index])
        right = float(amplitude[peak_index + 1]) if peak_index < amplitude.size - 1 else -np.inf
        if center >= threshold and center >= left and center >= right:
            local_peak_indices.append(peak_index)

    # If the relative threshold is too strict for a noisy profile, fall back to the strongest valid point.
    if not local_peak_indices:
        local_peak_indices = [int(valid_indices[int(np.argmax(amplitude[valid_indices]))])]

    # k가 작은 순서부터 확인하여 기본 주기에 가까운 후보를 우선 선택한다.
    for peak_index in sorted(local_peak_indices, key=lambda idx: float(ks[idx])):
        k, amplitude_value = refine_profile_peak(ks, amplitude, peak_index)
        if any(abs(k - peak.k) < AXIS_DUPLICATE_K_THRESHOLD for peak in peaks):
            continue
        peaks.append(
            AxisPeriodPeak(
                axis=axis,
                k=k,
                period_px=float(image_size) / k if k > 0 else float("inf"),
                amplitude=amplitude_value,
                source="amplitude",
            )
        )
        if len(peaks) >= count:
            # 호출자가 요청한 개수만큼 모으면 탐색을 종료한다.
            break
    return peaks




def candidates_from_axis_period_peaks(
    mag: np.ndarray,
    peaks_x: List[AxisPeriodPeak],
    peaks_y: List[AxisPeriodPeak],
    max_candidates: int,
) -> List[FftPeakCandidate]:
    # Step 7: Convert the selected x/y axis peaks into two axis-only candidates:
    # (x1, 0), (0, y1).
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    candidates: List[FftPeakCandidate] = []

    def add_candidate(kx: float, ky: float, score: float) -> None:
        # Step 8: Convert frequency k into a spatial period vector dx, dy.
        # 축 하나만 사용하는 후보에서는 다른 축의 k가 0이 된다.
        if len(candidates) >= max_candidates:
            return
        denom = (kx / float(w)) ** 2 + (ky / float(h)) ** 2
        if denom <= 0:
            return

        # 정규화된 주파수 벡터를 실제 픽셀 단위의 주기 벡터로 바꾼다.
        dx = (kx / float(w)) / denom
        dy = (ky / float(h)) / denom
        period = math.hypot(dx, dy)
        peak_x = float(cx) + kx
        peak_y = float(cy) + ky

        candidates.append(
            FftPeakCandidate(
                candidate_id=len(candidates) + 1,
                peak_x=peak_x,
                peak_y=peak_y,
                kx=kx,
                ky=ky,
                dx=dx,
                dy=dy,
                period_px=period,
                angle_deg=math.degrees(math.atan2(dy, dx)),
                candidate_score=float(score),
            )
        )

    if len(peaks_x) < 1 or len(peaks_y) < 1:
        # x와 y 중 한 축의 기본 피크라도 없으면 축 후보를 만들지 않는다.
        return candidates

    # 가장 강한 x 주기와 y 주기를 각각 단독 후보로 만든다.
    x1 = peaks_x[0]
    y1 = peaks_y[0]
    combos = [
        (x1, None),
        (None, y1),
    ]
    for peak_x_axis, peak_y_axis in combos:
        kx = round(peak_x_axis.k, 1) if peak_x_axis else 0.0
        ky = round(peak_y_axis.k, 1) if peak_y_axis else 0.0
        score = 0.0
        if peak_x_axis:
            score += peak_x_axis.amplitude
        if peak_y_axis:
            score += peak_y_axis.amplitude
        # 후보 점수는 선택된 축 피크의 진폭 합으로 기록한다.
        add_candidate(kx, ky, score)
    return candidates



def center_patch_box(image: Image.Image, patch_w: int, patch_h: int) -> Tuple[int, int, int, int]:
    # Step 12a: Locate a center-aligned patch with the requested period-scaled size.
    w, h = image.size
    patch_w = max(1, min(w, patch_w))
    patch_h = max(1, min(h, patch_h))
    cx = w / 2.0
    cy = h / 2.0
    left = int(round(cx - patch_w / 2.0))
    top = int(round(cy - patch_h / 2.0))
    right = left + patch_w
    bottom = top + patch_h

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > w:
        left -= right - w
        right = w
    if bottom > h:
        top -= bottom - h
        bottom = h

    return left, top, right, bottom


def sliding_window_sum(values: np.ndarray, window_h: int, window_w: int) -> np.ndarray:
    integral = np.pad(values, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (
        integral[window_h:, window_w:]
        - integral[:-window_h, window_w:]
        - integral[window_h:, :-window_w]
        + integral[:-window_h, :-window_w]
    )


def valid_cross_correlation_2d(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    h, w = template.shape[:2]
    full_shape = (source.shape[0] + h - 1, source.shape[1] + w - 1)
    corr = np.zeros((source.shape[0] - h + 1, source.shape[1] - w + 1), dtype=np.float64)
    src_fft = np.fft.rfftn(source, s=full_shape, axes=(0, 1))
    tpl_fft = np.fft.rfftn(template[::-1, ::-1], s=full_shape, axes=(0, 1))
    conv = np.fft.irfftn(src_fft * tpl_fft, s=full_shape, axes=(0, 1))
    corr[:, :] = conv[h - 1 : source.shape[0], w - 1 : source.shape[1]]
    return corr



def suppress_self_overlap(
    values: np.ndarray,
    origin_left: int,
    origin_top: int,
    patch_w: int,
    patch_h: int,
    fill_value: float,
    min_overlap_ratio: float = 0.95,
) -> np.ndarray:
    masked = values.copy()
    map_h, map_w = masked.shape
    if patch_w <= 0 or patch_h <= 0 or map_h == 0 or map_w == 0:
        return masked

    origin_right = origin_left + patch_w
    origin_bottom = origin_top + patch_h
    patch_area = float(patch_w * patch_h)

    xs = np.arange(map_w)
    ys = np.arange(map_h)
    overlap_w = np.maximum(
        0,
        np.minimum(origin_right, xs + patch_w) - np.maximum(origin_left, xs),
    )
    overlap_h = np.maximum(
        0,
        np.minimum(origin_bottom, ys + patch_h) - np.maximum(origin_top, ys),
    )
    overlap_ratio = np.outer(overlap_h, overlap_w) / patch_area
    masked[overlap_ratio >= min_overlap_ratio] = fill_value
    return masked



def basic_ncc(first: Image.Image, second: Image.Image) -> float:
    """Basic NCC over all RGB samples, without mean subtraction."""
    if first.size != second.size:
        raise ValueError("NCC inputs must have the same size")
    a = np.asarray(first.convert("RGB"), dtype=np.float64) / 255.0
    b = np.asarray(second.convert("RGB"), dtype=np.float64) / 255.0
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.clip(np.sum(a * b) / denom, -1.0, 1.0)) if denom > 1e-12 else float("nan")


def extend_patch_edges(patch: Image.Image) -> Image.Image:
    """Copy opposite edges in top/sides/bottom order to exactly double each axis."""
    pixels = np.asarray(patch.convert("RGB"))
    h, w = pixels.shape[:2]
    pad_x, pad_y = w // 2, h // 2
    pad_right, pad_bottom = w - pad_x, h - pad_y
    # Top center: copy the bottom edge of the original patch.
    top = pixels[np.arange(-pad_y, 0) % h]
    with_top = np.concatenate((top, pixels), axis=0)
    # Sides include the new top strip, filling both upper corners.
    left = with_top[:, np.arange(-pad_x, 0) % w]
    right = with_top[:, np.arange(pad_right) % w]
    with_sides = np.concatenate((left, with_top, right), axis=1)
    # Bottom: copy the original top rows across the full extended width.
    bottom = with_sides[pad_y + np.arange(pad_bottom) % h]
    return Image.fromarray(np.concatenate((with_sides, bottom), axis=0))


def save_period_patches(
    image: Image.Image,
    candidates: List[FftPeakCandidate],
    output_dir: str,
) -> None:
    # Step 12c: Make 16 center patches from 1x..4x of the detected x/y periods.
    ensure_dir(output_dir)
    x_candidates = [cand for cand in candidates if abs(cand.dx) > 1e-6 and abs(cand.dy) < 1e-6]
    y_candidates = [cand for cand in candidates if abs(cand.dy) > 1e-6 and abs(cand.dx) < 1e-6]
    if not x_candidates or not y_candidates:
        return

    base_dx = abs(x_candidates[0].dx)
    base_dy = abs(y_candidates[0].dy)
    score_rows = []
    montage_items: List[Tuple[int, int, Image.Image, Image.Image, Image.Image, Image.Image, float]] = []
    patch_sizes: List[Tuple[int, int, int, int]] = []
    for dy_mult in range(1, 5):
        for dx_mult in range(1, 5):
            patch_w = int(round(base_dx * dx_mult))
            patch_h = int(round(base_dy * dy_mult))
            left, top, right, bottom = center_patch_box(image, patch_w, patch_h)
            patch = image.crop((left, top, right, bottom))
            # Split odd sizes with the extra pixel on the right/bottom.
            pad_x, pad_y = patch.width // 2, patch.height // 2
            pad_right, pad_bottom = patch.width - pad_x, patch.height - pad_y
            grid_preview = draw_grid_patch_preview(image, base_dx, base_dy, (left, top, right, bottom))
            extended = extend_patch_edges(patch)
            patch.save(os.path.join(output_dir, f"period_patch_dx{dx_mult}_dy{dy_mult}.png"))
            # Match the exact source coordinates of the extended patch.
            box = (left - pad_x, top - pad_y, right + pad_right, bottom + pad_bottom)
            fits = box[0] >= 0 and box[1] >= 0 and box[2] <= image.width and box[3] <= image.height
            reference = image.crop((max(0, box[0]), max(0, box[1]),
                                    min(image.width, box[2]), min(image.height, box[3])))
            score = basic_ncc(extended, reference) if fits else float("nan")
            status = "ok" if np.isfinite(score) else ("zero_energy" if fits else "outside_image")
            score_rows.append(dict(dx_mult=dx_mult, dy_mult=dy_mult, patch_width=patch.width,
                                   patch_height=patch.height, pad_x=pad_x, pad_y=pad_y,
                                   pad_right=pad_right, pad_bottom=pad_bottom,
                                   comparison_width=extended.width, comparison_height=extended.height,
                                   ncc_score=score, status=status, rank=""))
            montage_items.append((dx_mult, dy_mult, grid_preview, patch, extended, reference, score))
            patch_sizes.append((dx_mult, dy_mult, patch.size[0], patch.size[1]))

    ranked = sorted((row for row in score_rows if row["status"] == "ok"),
                    key=lambda row: row["ncc_score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        print(f"patch rank={rank} dx={row['dx_mult']} dy={row['dy_mult']} NCC={row['ncc_score']:.6f}")
    with open(os.path.join(output_dir, "period_patch_scores.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(ranked + [row for row in score_rows if row["status"] != "ok"])
    save_patch_box_preview(image, patch_sizes, os.path.join(output_dir, "period_patch_boxes.png"))
    save_patch_montage(base_dx, base_dy, montage_items, os.path.join(output_dir, "period_patches_16.png"))


def main() -> None:
    # Step 14: CLI entry point. Runs the current axis-separated 1D FFT pipeline end to end.
    parser = argparse.ArgumentParser(description="Show RGB FFT center-to-peak period extraction on the full image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output-dir", default="outputs/fft_center_peak_example", help="Output directory.")
    parser.add_argument("--patch-output-dir", default="outputs/fft_patch", help="Output directory for period patches.")
    parser.add_argument("--min-period", type=float, default=10.0, help="Minimum spatial period.")
    parser.add_argument("--max-period", type=float, default=250.0, help="Maximum spatial period.")
    parser.add_argument("--max-peaks", type=int, default=20, help="Maximum center-to-peak candidates to keep.")
    parser.add_argument("--min-peak-percentile", type=float, default=80.0, help="Spectrum local maxima percentile threshold.")
    parser.add_argument("--dedupe-period-px", type=float, default=4.0, help="Minimum distance between duplicate spatial vectors.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    # Step 14a: Load input and build the frequency-domain visualization background.
    image, rgb = load_rgb(args.image)
    mag = fft_rgb_log_magnitude(rgb)
    period_min = args.min_period
    period_max = args.max_period

    # Step 14b: Build 1D FFT axis profiles, then hand only visualization to the draw module.
    ks, amp_x, amp_y, axis_peaks_x, axis_peaks_y = axis_period_profiles(
        rgb,
        min_period=period_min,
        max_period=period_max,
    )
    save_axis_period_functions(
        ks,
        amp_x,
        amp_y,
        axis_peaks_x,
        axis_peaks_y,
        os.path.join(args.output_dir, "axis_period_functions.png"),
        os.path.basename(args.image),
    )

    # Step 14c: Convert axis peaks into final candidates.
    candidates = candidates_from_axis_period_peaks(
        mag,
        axis_peaks_x,
        axis_peaks_y,
        max_candidates=min(args.max_peaks, 2),
    )
    visual_limit = 2


    # Step 14d: Write all visual and tabular outputs.
    draw_frequency_domain_peaks(
        mag,
        candidates,
        os.path.join(args.output_dir, "frequency_domain_peaks.png"),
        visual_limit=visual_limit,
    )
    draw_spatial_domain_peaks(
        image,
        candidates,
        os.path.join(args.output_dir, "spatial_domain_peaks.png"),
        visual_limit=visual_limit,
    )
    save_period_patches(image, candidates, args.patch_output_dir)

    print(f"output_dir: {args.output_dir}")
    print(f"image_size: width={image.size[0]}, height={image.size[1]}")
    print(f"spectrum_peaks_found: {len(candidates)}")
    for idx, peak in enumerate(axis_peaks_x, start=1):
        print(f"axis_x_peak_#{idx}: k={peak.k:.1f} period={peak.period_px:.1f}px amplitude={peak.amplitude:.3f}")
    for idx, peak in enumerate(axis_peaks_y, start=1):
        print(f"axis_y_peak_#{idx}: k={peak.k:.1f} period={peak.period_px:.1f}px amplitude={peak.amplitude:.3f}")
    for cand in candidates[:10]:
        print(
            f"#{cand.candidate_id:03d} k=({cand.kx:.1f},{cand.ky:.1f}) "
            f"dx={cand.dx:.2f} dy={cand.dy:.2f} "
            f"period={cand.period_px:.2f}px angle={cand.angle_deg:.1f} "
            f"score={cand.candidate_score:.3f}"
        )


if __name__ == "__main__":
    main()
