from __future__ import annotations

import argparse
import math
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np  
from PIL import Image

from period_candidates_visualization import (
    format_truncated,
    #draw_frequency_domain_peaks,
    draw_spatial_domain_peaks,
    save_axis_period_functions,
    save_period_patches,
)

#cosine 
AXIS_PROFILE_MAX_K = 100
AXIS_PEAK_RELATIVE_THRESHOLD = 0.5
min_period = 1024.0 / 100.0
max_period = 1024.0 / 3.0


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

    vote_x = vote_axis_peaks("x", ks, [amp_0x, amp_1x, amp_2x], valid_x, w)
    vote_y = vote_axis_peaks("y", ks, [amp_0y, amp_1y, amp_2y], valid_y, h)
    result = (ks, vote_x.amplitude, vote_y.amplitude, vote_x.peaks, vote_y.peaks)
    return (*result, vote_x, vote_y) if include_votes else result


@dataclass
class AxisPeakVote:
    peaks: List[AxisPeriodPeak]
    amplitude: np.ndarray


def vote_axis_peaks(axis, ks, amplitudes, valid, image_size) -> AxisPeakVote:
    """Vote by the selected FFT bin, independent of channel amplitudes.

    Retain each channel's subpixel estimate. The first agreeing channel supplies
    the majority estimate. With three distinct bins, use the strongest peak.
    """
    peaks = [select_axis_amplitude_peaks(axis, ks, amp, valid, image_size)
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
    """유효 최댓값의 threshold 이상인 peaks 모두."""
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return []
    if not np.any(np.isfinite(amplitude[indices]) & (amplitude[indices] > 0)):
        return []
    peak_index = []
    threshold = AXIS_PEAK_RELATIVE_THRESHOLD * float(np.max(amplitude[indices]))
    max_r = 0.0
    final_index = -1
    for index in sorted(indices, key=lambda i: float(ks[i])):
        left = amplitude[index - 1] if index > 0 else -np.inf
        right = amplitude[index + 1] if index < amplitude.size - 1 else -np.inf
        if amplitude[index] >= threshold and amplitude[index] >= max(left, right):
            peak_index.append(index)
    for index in peak_index:
        h = amplitude[index]
        w = ks[index]
        r = float(np.sqrt(h**2 + w**2))
        if r > max_r:
            max_r = r
            final_index = index

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
def _vgg19_conv2_2_model():
    from torchvision import models
    # Reuse the same pretrained model for the original and all tiled candidates.
    return models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:9].eval()


def extract_vgg19_features(image: Image.Image) -> dict:
    import torch
    from torchvision import transforms
    with torch.no_grad():
        tensor = transforms.ToTensor()(image.convert("RGB")).unsqueeze(0)
        return {"conv2_2": _vgg19_conv2_2_model()(tensor)}


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


def vgg19(image: Image.Image, output_dir: str) -> dict:
    import torch
    import matplotlib.pyplot as plt
    from torchvision import models, transforms

    # 1. 저장 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    feature_maps = extract_vgg19_features(image)

    # 6. Feature Map 크기 출력
    for name, fmap in feature_maps.items():
        print(name, fmap.shape)

    # # 7. 시각화
    # fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    # for row, (name, fmap) in enumerate(feature_maps.items()):
    #     channels = fmap.squeeze(0).cpu().numpy()
    #     for col in range(8):
    #         axes[row, col].imshow(channels[col], cmap="gray")
    #         axes[row, col].axis("off")
    #     axes[row, 0].set_title(name)
    # plt.tight_layout()

    # # 8. 시각화 이미지 저장
    # image_path = os.path.join(output_dir, "feature_maps.png")
    # plt.savefig(image_path, dpi=200)

    # # 9. Feature Map Tensor 저장
    # tensor_path = os.path.join(output_dir, "feature_maps.pt")
    # torch.save(feature_maps, tensor_path)
    return feature_maps


def vgg19_comparison_mask(image_size, feature_shape, patch_box) -> np.ndarray:
    """Build a boolean mask directly in feature space, excluding the center.

    Floor left/top and ceil right/bottom to exclude every feature cell whose
    spatial bin overlaps the central patch. No mask resize or partial weights.
    """
    width, height = image_size
    fh, fw = feature_shape
    left, top, right, bottom = patch_box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Patch box must be inside original image")
    mask = np.ones((fh, fw), dtype=bool)
    mask[math.floor(top * fh / height):math.ceil(bottom * fh / height),
         math.floor(left * fw / width):math.ceil(right * fw / width)] = False
    return mask


def vgg19_score(feature_maps: dict, patch: Image.Image, image: Image.Image,
                patch_box: Tuple[int, int, int, int],
                tiled: Image.Image | None = None) -> float:
    """Mean spatial feature cosine outside the central patch; higher is better."""
    import torch
    if tiled is None:
        tiled = tile_patch_to_image(patch, image.size, patch_box)
    original_features = feature_maps["conv2_2"]
    mask = vgg19_comparison_mask(image.size, original_features.shape[2:], patch_box)
    if not mask.any():
        return float("nan")
    with torch.no_grad():
        tiled_features = extract_vgg19_features(tiled)["conv2_2"]
        if original_features.shape != tiled_features.shape:
            raise ValueError("Original and tiled conv2_2 shapes must match")
        selected = torch.as_tensor(mask, device=tiled_features.device)
        original = original_features.to(tiled_features.device)[0, :, selected]
        candidate = tiled_features[0, :, selected]
        # Keep the existing exclusion of channels with no response in either image.
        active = torch.any(original != 0, dim=1) | torch.any(candidate != 0, dim=1)
        if not torch.any(active):
            return float("nan")
        original = original[active]
        candidate = candidate[active]
        # [C, N]: compare channel vectors at each position, then average.
        return torch.nn.functional.cosine_similarity(
            original, candidate, dim=0, eps=1e-8
        ).mean().item()


def create_period_patches(
    image: Image.Image,
    candidates: List[FftPeakCandidate],
    score_method: str = "ncc",
    feature_maps: dict | None = None,
) -> List[PatchResult]:
    # Step 12c: Make 16 center patches from 1x..4x of the detected x/y periods.
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
    # Add one base period regardless of the patch multiplier; odd pixels go right/bottom.
    extra_w, extra_h = max(1, round(base_dx)), max(1, round(base_dy))
    
    results: List[PatchResult] = []
    for dy_mult in range(1, 5):
        for dx_mult in range(1, 5):
            patch_w = int(round(base_dx * dx_mult))
            patch_h = int(round(base_dy * dy_mult))
            # patch사이즈를 1부터 이미지 사이즈까지로 벗어나지 않게 조정
            patch_w, patch_h = max(1, min(w, patch_w)), max(1, min(h, patch_h))
            left, top =round((w - patch_w) / 2), round((h - patch_h) / 2)
            right, bottom = left + patch_w, top + patch_h

            ''' image_patch, extended, reference '''
            image_patch = image.crop((left, top, right, bottom))
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
                                    (left, top, right, bottom), tiled)
                status = "ok" if np.isfinite(score) else "no_comparison_signal"
            else:
                score = (opencv_ncc(extended, reference, margin_mask)
                         if reference is not None else float("nan"))
                status = "ok" if np.isfinite(score) else ("zero_energy" if fits else "outside_image")
            metadata = dict(dx = base_dx, dy = base_dy, dx_mult=dx_mult, dy_mult=dy_mult, patch_width=image_patch.width,
                                   patch_height=image_patch.height,
                                   comparison_width=extended.width, comparison_height=extended.height,
                                   score_method=score_method, status=status, rank="")
            metadata[score_key] = score
            if score_method == "vgg19":
                metadata.update(comparison_width=w, comparison_height=h, score_metric="spatial_feature_cosine_similarity")
            results.append(PatchResult(image_patch, extended, reference, metadata, tiled))

    results.sort(key=lambda r: (
        not np.isfinite(r.metadata[score_key]),
        -r.metadata[score_key]
        if np.isfinite(r.metadata[score_key]) else float("inf")))
    for rank, result in enumerate(results, start=1):
        row = result.metadata
        if row["status"] == "ok":
            row["rank"] = rank
  

 
    return results


def select_final_periods(image, vote_x, vote_y, score_method="ncc", feature_maps=None):
    """Keep each axis vote winner regardless of subsequent patch scores."""
    candidates = candidates_from_axis_period_peaks(
        (image.height, image.width), vote_x.peaks, vote_y.peaks)
    results = create_period_patches(image, candidates, score_method, feature_maps)
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
    args = parser.parse_args()
   

    ensure_dir(args.output_dir)
    # Step 14a: Load input and build the frequency-domain visualization background.
    image, rgb = load_rgb(args.image)
    # mag = fft_rgb_log_magnitude(rgb)

    fft_option = args.fout_opt.lower()
    patch_option = args.pout_opt.lower()

    feature_maps = None
    if args.score_method == "vgg19":
        feature_maps = vgg19(image, args.output_dir)

    # Step 14b: Build 1D FFT axis profiles, then hand only visualization to the draw module.
    ks, amp_x, amp_y, axis_peaks_x, axis_peaks_y, vote_x, vote_y = axis_period_profiles(
        rgb, include_votes=True)
    amp_x, amp_y, axis_peaks_x, axis_peaks_y, candidates, results = select_final_periods(
        image, vote_x, vote_y, args.score_method, feature_maps)
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
            os.path.join(args.output_dir, "spatial_domain_peaks.png"),
            visual_limit=visual_limit,
        )
    if patch_option == "off":
        print("Patch output is disabled. Skipping patch image output.")
    else:
        save_period_patches(image, results, args.output_dir, args.score_method)

    print(f"output_dir: {args.output_dir}")
    print(f"image_size: width={image.size[0]}, height={image.size[1]}")

    for idx, peak in enumerate(axis_peaks_x, start=1):
        print(f"axis_x_peak_#{idx}: k={format_truncated(peak.k)} period={format_truncated(peak.period_px)}px amplitude={peak.amplitude:.3f}")
    for idx, peak in enumerate(axis_peaks_y, start=1):
        print(f"axis_y_peak_#{idx}: k={format_truncated(peak.k)} period={format_truncated(peak.period_px)}px amplitude={peak.amplitude:.3f}")
   



if __name__ == "__main__":
    main()
