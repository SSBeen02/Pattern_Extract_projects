from __future__ import annotations

# VGG19 features의 0-based convolution 인덱스. 아래 값만 바꾸면 됩니다.
# conv1_1=0,  conv1_2=2
# conv2_1=5,  conv2_2=7
# conv3_1=10, conv3_2=12, conv3_3=14, conv3_4=16
# conv4_1=19, conv4_2=21, conv4_3=23, conv4_4=25
# conv5_1=28, conv5_2=30, conv5_3=32, conv5_4=34
# 각 convolution 다음 인덱스는 ReLU입니다. 기존처럼 ReLU까지 적용합니다.
VGG_CONV_LAYER_INDEX = 16

import argparse
import math
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np  
from PIL import Image
from period_output_naming import image_id, save_ranked_patches

from period_candidates_visualization import (
    format_truncated,
    #draw_frequency_domain_peaks,
    draw_spatial_domain_peaks,
    save_axis_period_functions,
    save_period_patches,
)
''' ncc or vgg: configured convolution layer or dense 선택'''
#cosine 
AXIS_PROFILE_MAX_K = 100
AXIS_PEAK_RELATIVE_THRESHOLD = 0.5
min_period = 1024.0 / 100.0
max_period = 1024.0 / 3.0
alpha = 0.1

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
    """평균을 제거하고 표준편차로 나눈다. 상수 신호는 나눗셈을 생략한다."""
    channel = channel - float(channel.mean())
    std = float(channel.std())
    if std > 1e-6:
        channel = channel / std
    return channel.astype(np.float32)


def axis_period_profiles(
    rgb: np.ndarray,
    max_k: int = AXIS_PROFILE_MAX_K,
    include_votes: bool = False,
    peak_selector: Callable | None = None,
):
    # Standardize RGB channels before FFT; spectra are not separately scaled.
    rgb = np.array([normalize_channel(channel) for channel in rgb.transpose(2, 0, 1)]).transpose(1, 2, 0)
    h, w, _channels = rgb.shape
    max_axis_k = min(w // 2, h // 2, max_k) # max_axis_k는 축별로 최대 k를 제한한다. FFT bin은 N/2까지 존재하므로 w//2, h//2를 고려한다.
    ks = np.arange(1, max_axis_k + 1, dtype=np.float32)

    # 전체 주파수 길이로 누적할 버퍼 (행/열 합산 결과가 들어감)
    acc_x = np.zeros(w // 2 + 1, dtype=np.float64)
    acc_y = np.zeros(h // 2 + 1, dtype=np.float64)

    window_x = np.hanning(w).astype(np.float32)
    window_y = np.hanning(h).astype(np.float32)
    amp_0x = np.zeros(max_axis_k, dtype=np.float64)
    amp_0y = np.zeros(max_axis_k, dtype=np.float64)
    amp_1x = np.zeros(max_axis_k, dtype=np.float64)
    amp_1y = np.zeros(max_axis_k, dtype=np.float64)
    amp_2x = np.zeros(max_axis_k, dtype=np.float64)
    amp_2y = np.zeros(max_axis_k, dtype=np.float64)
    for channel_idx in range(3):
        channel = rgb[:, :, channel_idx].astype(np.float64)

        # x 방향
        signal_x = channel - channel.mean(axis=1, keepdims=True)
        signal_x *= window_x[None, :]
        spectrum_x = np.fft.rfft(signal_x, axis=1)      # (h, w//2+1)
        power_x = np.abs(spectrum_x).sum(axis=0)        # (w//2+1,) 행 방향 합산

        # y 방향
        signal_y = channel - channel.mean(axis=0, keepdims=True)
        signal_y *= window_y[:, None]
        spectrum_y = np.fft.rfft(signal_y, axis=0)      # (h//2+1, w)
        power_y = np.abs(spectrum_y).sum(axis=1)        # (h//2+1,) 열 방향 합산

        # DC 제외, 1..max_axis_k 구간만 누적
        if channel_idx == 0:
            amp_0x = power_x[1 : max_axis_k + 1].copy()
        if channel_idx == 0:
            amp_0y = power_y[1 : max_axis_k + 1].copy()
        if channel_idx == 1:
            amp_1x = power_x[1 : max_axis_k + 1].copy()
        if channel_idx == 1:
            amp_1y = power_y[1 : max_axis_k + 1].copy()
        if channel_idx == 2:
            amp_2x = power_x[1 : max_axis_k + 1].copy()
        if channel_idx == 2:
            amp_2y = power_y[1 : max_axis_k + 1].copy()

    # # 채널 3개로 한 번에 나눔 
    # amp_x /= (3.0)
    # amp_y /= (3.0)
    
    
    # Step 5: Apply the spatial-period limits as a valid k range.
    # 이미지 길이 N에서 k cycles가 나타나면 한 주기는 N/k 픽셀이다.
    low_freq_cut_x = float(math.ceil(w / max_period))
    low_freq_cut_y = float(math.ceil(h / max_period))
 

    valid_x = (ks >= low_freq_cut_x) & (ks <= AXIS_PROFILE_MAX_K) #유효한 주파수 범위에 해당하는 FFT bin만 True로 표시
    valid_y = (ks >= low_freq_cut_y) & (ks <= AXIS_PROFILE_MAX_K)
    if not np.any(valid_x):
        # 설정 범위가 FFT bin과 겹치지 않으면 전체 계산 범위를 fallback으로 사용한다.
        valid_x = np.ones(ks.shape, dtype=bool)
    if not np.any(valid_y):
        valid_y = np.ones(ks.shape, dtype=bool)

    vote_x = vote_axis_peaks("x", ks, [amp_0x, amp_1x, amp_2x], valid_x, w, peak_selector)
    vote_y = vote_axis_peaks("y", ks, [amp_0y, amp_1y, amp_2y], valid_y, h, peak_selector)
    result = (ks, vote_x.amplitude, vote_y.amplitude, vote_x.peaks, vote_y.peaks)
    return (*result, vote_x, vote_y) if include_votes else result


@dataclass
class AxisPeakVote:
    peaks: List[AxisPeriodPeak]
    amplitude: np.ndarray


def vote_axis_peaks(axis, ks, amplitudes, valid, image_size, peak_selector=None) -> AxisPeakVote:
    """Vote by the selected FFT bin, independent of channel amplitudes.

    Retain each channel's subpixel estimate. The first agreeing channel supplies
    the majority estimate. With three distinct bins, use the strongest peak.
    """
    selector = peak_selector or select_axis_amplitude_peaks
    peaks = [selector(axis, ks, amp, valid, image_size)
             for amp in amplitudes]
    groups = {}
    for channel, selected in enumerate(peaks):
        if selected:
            bin_index = int(np.argmin(np.abs(ks - selected[0].k)))
            groups.setdefault(bin_index, []).append(channel)
    if not groups:
        return AxisPeakVote([], amplitudes[0])
    majority = max(groups.values(), key=len)
    if len(majority) >= 2:
        winner = majority[0]
    else:
        winner = max((i for group in groups.values() for i in group),
                     key=lambda i: peaks[i][0].amplitude)
    return AxisPeakVote(peaks[winner], amplitudes[winner])



def subpixel_peak_1d(v_minus: float, v_zero: float, v_plus: float) -> float:
    # Step 6a: Estimate where the peak lies between integer k samples with a parabola.
    # 세 샘플을 지나는 포물선의 꼭짓점 위치를 계산한다.
    if not (v_zero >= v_minus and v_zero >= v_plus):
        # 국소 최대가 아니면 정수 위치를 유지한다.
        return 0.0
    denom = v_minus - 2.0 * v_zero + v_plus
    if not (denom < 0.0) or abs(denom) < 1e-9:
        # 곡률이 음수가 아니거나 거의 평평하면 정수 위치를 유지한다.
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
    k = float(ks[peak_index]) + offset
    value = float(values[peak_index])
    return k, value


def select_axis_amplitude_peaks(
    axis: str, ks: np.ndarray, amplitude: np.ndarray,
    valid: np.ndarray, image_size: int,
) -> List[AxisPeriodPeak]:
    """Select the farthest local peak passing an origin-based threshold."""
    indices = np.flatnonzero(valid & np.isfinite(ks) & (ks > 0))
    if indices.size == 0:
        return []
    if not np.any(np.isfinite(amplitude[indices]) & (amplitude[indices] > 0)):
        return []
    finite_indices = indices[np.isfinite(amplitude[indices])]
    strongest = max(finite_indices, key=lambda i: float(amplitude[i]))
    max_amplitude = float(amplitude[strongest])

    peak_indices = []

    for index in indices:
        k = float(ks[index])
        if k <= 0:
            continue
        threshold = max_amplitude / (k ** alpha)
        left = amplitude[index - 1] if index > 0 else -np.inf
        right = amplitude[index + 1] if index < amplitude.size - 1 else -np.inf
        if np.isfinite(amplitude[index]) and amplitude[index] >= threshold and amplitude[index] >= max(left, right):
            peak_indices.append(index)
    final_index = max(peak_indices, key=lambda i: float(ks[i]), default=-1)

    if final_index < 0:
        return []
    k, value = refine_profile_peak(ks, amplitude, final_index)
    return [AxisPeriodPeak(axis, k, image_size / k if k > 0 else float("inf"), value, "amplitude")]


def candidates_from_axis_period_peaks(
    image_shape: Tuple[int, int], peaks_x: List[AxisPeriodPeak], peaks_y: List[AxisPeriodPeak],
) -> List[FftPeakCandidate]:
    """축별 기본 피크를 x/y 공간 주기 후보로 변환한다."""
    if not peaks_x or not peaks_y:
        return []
    h, w = image_shape
    candidates = []
    for axis, peak, size in (("x", peaks_x[0], w), ("y", peaks_y[0], h)):
        k = float(peak.k) #달라진 포인트. round적용 안함
        period = size / k
        kx, ky = (k, 0.0) if axis == "x" else (0.0, k)
        dx, dy = (period, 0.0) if axis == "x" else (0.0, period)
        candidates.append(FftPeakCandidate(
            candidate_id=len(candidates) + 1, peak_x=w // 2 + kx, peak_y=h // 2 + ky,
            kx=kx, ky=ky, dx=dx, dy=dy, period_px=period,
            angle_deg=0.0 if axis == "x" else 90.0, candidate_score=float(peak.amplitude),
        ))
    return candidates

#-------------------------------------------------------
# patch
#-------------------------------------------------------

def extend_patch_edges(image_patch: Image.Image, extra_w: int, extra_h: int) -> Image.Image:
    """기본 주기 한 개를 좌우·상하에 나눠 반대쪽 가장자리로 확장한다."""
    pixels = np.asarray(image_patch.convert("RGB"))
    return Image.fromarray(np.pad(
        pixels, ((extra_h // 2, extra_h - extra_h // 2),
                 (extra_w // 2, extra_w - extra_w // 2), (0, 0)), mode="wrap",
    ))

@ dataclass
class PatchResult:
    image_patch: Image.Image
    extended: Image.Image
    reference: Image.Image | None
    metadata: dict
    tiled: Image.Image | None = None

@ dataclass
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

def opencv_ncc(first: Image.Image, second: Image.Image, mask: np.ndarray = None) -> float:
    """Channel-centered pooled NCC (TM_CCOEFF_NORMED formula)."""
    if first.size != second.size:
        raise ValueError("NCC inputs must have the same size")
    a = np.asarray(first.convert("RGB"), dtype=np.float64) / 255.0  
    b = np.asarray(second.convert("RGB"), dtype=np.float64) / 255.0

    if mask is not None:
        if mask.shape != a.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("NCC mask must be boolean and match the image size")
        if not np.any(mask):
            return float("nan")
        # 마진만 선택한 뒤 기존 채널별 평균 제거 및 pooled NCC를 적용한다.
        a = a[mask][:, None, :]
        b = b[mask][:, None, :]

    a = a - a.mean(axis=(0, 1), keepdims=True)
    b = b - b.mean(axis=(0, 1), keepdims=True)
    num = np.sum(a * b)
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.clip(num / denom, -1.0, 1.0)) if denom > 1e-12 else float("nan")


@lru_cache(maxsize=1)
def _vgg19_dense_model():
    from torchvision import models
    # Keep the full model available for convolutional and FC1 extraction.
    return models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()


def vgg_conv_layer_name() -> str:
    layers = {
        0: "conv1_1", 2: "conv1_2", 5: "conv2_1", 7: "conv2_2",
        10: "conv3_1", 12: "conv3_2", 14: "conv3_3", 16: "conv3_4",
        19: "conv4_1", 21: "conv4_2", 23: "conv4_3", 25: "conv4_4",
        28: "conv5_1", 30: "conv5_2", 32: "conv5_3", 34: "conv5_4",
    }
    if VGG_CONV_LAYER_INDEX not in layers:
        raise ValueError(f"Invalid VGG_CONV_LAYER_INDEX: {VGG_CONV_LAYER_INDEX}; choose from {tuple(layers)}")
    return layers[VGG_CONV_LAYER_INDEX]


def extract_vgg19_features(image: Image.Image, feature_mode: str = "dense") -> dict:
    """Extract FC1 ReLU outputs or configured conv activations without saving feature images."""
    import torch
    from torchvision import transforms
    if feature_mode not in ("conv", "dense"):
        raise ValueError("feature_mode must be conv or dense")
    with torch.no_grad():
        preprocessing = ([transforms.Resize((224, 224))] if feature_mode == "dense" else [])
        tensor = transforms.Compose(preprocessing + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(image.convert("RGB")).unsqueeze(0)
        model = _vgg19_dense_model()
        if feature_mode == "conv":
            layer_name = vgg_conv_layer_name()
            return {layer_name: model.features[:VGG_CONV_LAYER_INDEX + 2](tensor)}
        features = model.avgpool(model.features(tensor))
        # classifier[:2] applies FC1 and ReLU (4096 outputs), before dropout.
        return {"dense_fc1": model.classifier[:2](torch.flatten(features, 1))}


def tile_patch_to_image(patch: Image.Image, image_size: Tuple[int, int],
                        patch_box: Tuple[int, int, int, int]) -> Image.Image:
    """Repeat RGB pixels with the original patch phase; do not stretch the motif."""
    left, top, right, bottom = patch_box
    width, height = image_size
    if (right - left, bottom - top) != patch.size:
        raise ValueError("Patch box must match patch size")
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Patch box must be inside the original image")
    pixels = np.asarray(patch.convert("RGB"))
    xs = (np.arange(width) - left) % patch.width
    ys = (np.arange(height) - top) % patch.height
    return Image.fromarray(pixels[ys[:, None], xs[None, :]])


def rank_period_patches(results, score_method, rank_small=False):
    """Descending score only; rank_small is an ignored compatibility argument."""
    key = f"{score_method}_score"
    results.sort(key=lambda r: (not np.isfinite(r.metadata[key]),
                               -r.metadata[key] if np.isfinite(r.metadata[key]) else 0))
    for rank, result in enumerate(results, start=1):
        row = result.metadata
        row["rank_small"] = False
        row["rank"] = rank if row["status"] == "ok" and np.isfinite(row[key]) else ""
    return results


def vgg19_score(feature_maps: dict, patch: Image.Image, image: Image.Image,
                patch_box: Tuple[int, int, int, int],
                tiled: Image.Image | None = None,
                feature_mode: str = "dense") -> float:
    """Dense cosine or mean per-position conv cosine, without spatial masking."""
    import torch
    if tiled is None:
        tiled = tile_patch_to_image(patch, image.size, patch_box)
    with torch.no_grad():
        tiled_maps = extract_vgg19_features(tiled, feature_mode)
        key = vgg_conv_layer_name() if feature_mode == "conv" else "dense_fc1"
        original, candidate = feature_maps[key], tiled_maps[key]
        if original.shape != candidate.shape:
            raise ValueError("Original and tiled feature shapes must match")
        if original.norm().item() == 0 or candidate.norm().item() == 0:
            return float("nan")
        return torch.nn.functional.cosine_similarity(
            original.to(candidate.device), candidate, dim=1, eps=1e-8
        ).mean().item()


def average_neighbor_patches(image: Image.Image, patch_box) -> Tuple[Image.Image, int]:
    """Average center plus four one-patch-away neighbors at matching pixel phases.

    Boundaries use only available image pixels, without padding.
    """
    left, top, right, bottom = patch_box
    pw, ph = right-left, bottom-top
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    total = np.zeros((ph, pw, 3), dtype=np.float64)
    counts = np.zeros((ph, pw, 1), dtype=np.float64)
    used = 0
    for dx, dy in ((0, 0), (-pw, 0), (pw, 0), (0, -ph), (0, ph)):
        x, y = left+dx, top+dy
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(image.width, x+pw), min(image.height, y+ph)
        if x0 >= x1 or y0 >= y1:
            continue
        dst = (slice(y0-y, y1-y), slice(x0-x, x1-x))
        total[dst] += pixels[y0:y1, x0:x1]
        counts[dst] += 1
        used += 1
    return Image.fromarray(np.rint(total/counts).clip(0, 255).astype(np.uint8)), used


def create_period_patches(
    image: Image.Image,
    candidates: List[FftPeakCandidate],
    score_method: str = "ncc",
    feature_maps: dict | None = None,
    rank_small: bool = False,
    vgg_feature: str = "dense",
) -> List[PatchResult]:
    # Make nine averaged representative patches (1x..3x).
    if score_method not in ("ncc", "vgg19"):
        raise ValueError("score_method must be ncc or vgg19")
    w, h = image.size
    if score_method == "vgg19":
        if feature_maps is None:
            raise ValueError("VGG19 scoring requires the original image feature maps")
    score_key = f"{score_method}_score"
    x_candidates = [cand for cand in candidates if abs(cand.dx) > 1e-6 and abs(cand.dy) < 1e-6]
    y_candidates = [cand for cand in candidates if abs(cand.dy) > 1e-6 and abs(cand.dx) < 1e-6]
    if not x_candidates or not y_candidates:
        return []

    base_dx = abs(x_candidates[0].dx)
    base_dy = abs(y_candidates[0].dy)
    base_patch_w = max(1, round(base_dx))
    base_patch_h = max(1, round(base_dy))
    # Add one base period regardless of the patch multiplier; odd pixels go right/bottom.
    extra_w, extra_h = base_patch_w, base_patch_h
    
    results: List[PatchResult] = []
    for dy_mult in range(1, 4):
        for dx_mult in range(1, 4):
            patch_w = base_patch_w * dx_mult
            patch_h = base_patch_h * dy_mult
            # patch사이즈를 1부터 이미지 사이즈까지로 벗어나지 않게 조정
            patch_w, patch_h = max(1, min(w, patch_w)), max(1, min(h, patch_h))
            left, top =round((w - patch_w) / 2), round((h - patch_h) / 2)
            right, bottom = left + patch_w, top + patch_h

            ''' image_patch, extended, reference '''
            # VGG uses the five-region average; NCC uses the unmodified center crop.
            averaged_patch, averaged_patch_count = average_neighbor_patches(image, (left, top, right, bottom))
            image_patch = (averaged_patch if score_method == "vgg19"
                           else image.crop((left, top, right, bottom)))
            extended = extend_patch_edges(image_patch, extra_w, extra_h)
          
            box = ( left - extra_w // 2, top - extra_h // 2, right + extra_w - extra_w // 2, bottom + extra_h - extra_h // 2)
            fits = box[0] >= 0 and box[1] >= 0 and box[2] <= image.width and box[3] <= image.height
            reference = image.crop(box) if fits else None

            margin_mask = np.ones((extended.height, extended.width), dtype=bool)
            margin_left, margin_top = extra_w // 2, extra_h // 2
            margin_mask[
                margin_top:margin_top + image_patch.height,
                margin_left:margin_left + image_patch.width,
            ] = False
            tiled = None
            if score_method == "vgg19":
                tiled = tile_patch_to_image(image_patch, image.size, (left, top, right, bottom))
                score = vgg19_score(feature_maps, image_patch, image,
                                    (left, top, right, bottom), tiled, feature_mode=vgg_feature)
                status = "ok" if np.isfinite(score) else "no_comparison_signal"
            else:
                score = (opencv_ncc(extended, reference, margin_mask)
                         if reference is not None else float("nan"))
                status = "ok" if np.isfinite(score) else ("zero_energy" if fits else "outside_image")
            metadata = dict(dx = base_dx, dy = base_dy, dx_mult=dx_mult, dy_mult=dy_mult, patch_width=image_patch.width,
                                   patch_height=image_patch.height,
                                   comparison_width=extended.width, comparison_height=extended.height,
                                   score_method=score_method, status=status, rank="")
            metadata["averaged_patch_count"] = averaged_patch_count
            metadata[score_key] = score
            if score_method == "vgg19":
                metadata.update(comparison_width=w, comparison_height=h,
                                vgg_feature=vgg_feature,
                                vgg_layer=(vgg_conv_layer_name() if vgg_feature == "conv" else "fc1"),
                                score_metric=(f"{vgg_conv_layer_name()}_spatial_cosine_similarity" if vgg_feature == "conv"
                                              else "fc1_cosine_similarity"))
            results.append(PatchResult(image_patch, extended, reference, metadata, tiled))

    return rank_period_patches(results, score_method, rank_small)


def select_final_periods(image, vote_x, vote_y, score_method="ncc", feature_maps=None,
                         rank_small=False, vgg_feature="dense"):
    """Keep each axis vote winner regardless of subsequent patch scores."""
    candidates = candidates_from_axis_period_peaks(
        (image.height, image.width), vote_x.peaks, vote_y.peaks)
    results = create_period_patches(image, candidates, score_method, feature_maps,
                                    rank_small=rank_small, vgg_feature=vgg_feature)
    return (vote_x.amplitude, vote_y.amplitude, vote_x.peaks, vote_y.peaks,
            candidates, results)


def main() -> None:
    # Step 14: CLI entry point. Runs the current axis-separated 1D FFT pipeline end to end.
    parser = argparse.ArgumentParser(description="Show RGB FFT center-to-peak period extraction on the full image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output_dir", default="outputs", help="Output directory.")
    parser.add_argument("--fout-opt", default="on", help="Choose whether to save FFT output.")
    parser.add_argument("--pout-opt", default="on", help="Choose whether to save patch output.")


    parser.add_argument("--score-method", choices=("ncc", "vgg19"), default="ncc",
                        help="Patch similarity method (default: ncc).")
    parser.add_argument("--vgg-feature", choices=("conv", "dense"), default="dense",
                        help=f"VGG19 features: {vgg_conv_layer_name()} full spatial map or FC1 ReLU outputs (default: dense).")
    parser.add_argument("--rank-small", action="store_true",
                        help="Deprecated compatibility option; ranking always uses similarity score only.")
    args = parser.parse_args()
    if args.rank_small and args.score_method != "vgg19":
        parser.error("--rank-small requires --score-method vgg19")
   

    ensure_dir(args.output_dir)
    # Step 14a: Load input and build the frequency-domain visualization background.
    image, rgb = load_rgb(args.image)
    # mag = fft_rgb_log_magnitude(rgb)

    fft_option = args.fout_opt.lower()
    patch_option = args.pout_opt.lower()

    feature_maps = (extract_vgg19_features(image, args.vgg_feature)
                    if args.score_method == "vgg19" else None)

    # Step 14b: Build 1D FFT axis profiles, then hand only visualization to the draw module.
    ks, amp_x, amp_y, axis_peaks_x, axis_peaks_y, vote_x, vote_y = axis_period_profiles(
        rgb, include_votes=True)
    amp_x, amp_y, axis_peaks_x, axis_peaks_y, candidates, results = select_final_periods(
        image, vote_x, vote_y, args.score_method,
        feature_maps=feature_maps, rank_small=args.rank_small, vgg_feature=args.vgg_feature)
    save_axis_period_functions(
        ks,
        amp_x,
        amp_y,
        axis_peaks_x,
        axis_peaks_y,
        os.path.join(args.output_dir, f"{image_id(args.image)}_axis.png"),
        os.path.basename(args.image),
    )

    # Step 14c: Convert axis peaks into final candidates.
    candidates = candidates_from_axis_period_peaks(
        rgb.shape[:2], axis_peaks_x, axis_peaks_y,
    )
    visual_limit = 2


    # Step 14d: Write all visual and tabular outputs.
    if fft_option == "off":
        print("FFT output is disabled. Skipping FFT visualizations.")
    else:
        draw_spatial_domain_peaks(
            image,
            candidates,
            os.path.join(args.output_dir, f"{image_id(args.image)}_period.png"),
            visual_limit=visual_limit,
        )
    if patch_option == "off":
        print("Patch output is disabled. Skipping patch image output.")
    else:
        save_ranked_patches(image, results, args.output_dir, args.score_method, args.image)

    print(f"output_dir: {args.output_dir}")
    print(f"image_size: width={image.size[0]}, height={image.size[1]}")

    for idx, peak in enumerate(axis_peaks_x, start=1):
        print(f"axis_x_peak_#{idx}: k={format_truncated(peak.k)} period={format_truncated(peak.period_px)}px amplitude={peak.amplitude:.3f}")
    for idx, peak in enumerate(axis_peaks_y, start=1):
        print(f"axis_y_peak_#{idx}: k={format_truncated(peak.k)} period={format_truncated(peak.period_px)}px amplitude={peak.amplitude:.3f}")
   



if __name__ == "__main__":
    main()
