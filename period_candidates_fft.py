import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from period_candidates_visualization import (
    draw_frequency_domain_peaks,
    draw_spatial_domain_peaks,
    save_axis_period_functions,
    save_patch_box_preview,
    safe_font,
)


AXIS_PROFILE_MAX_K = 100
AXIS_PEAK_RELATIVE_THRESHOLD = 0.73
MIN_PERIOD = 10.0
MAX_PERIOD = 256.0
#k range is 4~100

# Period candidates shared by visualization and patch extraction.
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


@dataclass
class PatchResult:
    image_patch: Image.Image
    extended: Image.Image
    reference: Image.Image | None
    metadata: dict


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


def normalize_01(values: np.ndarray) -> np.ndarray:
    """유한한 최솟값·최댓값으로 0~1 정규화하며 상수 배열은 0으로 만든다."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[AxisPeriodPeak], List[AxisPeriodPeak]]:
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

    # 기준을 통과하는 가장 작은 k의 피크를 축별로 선택한다.
    peaks_x = select_axis_amplitude_peaks("x", ks, amp_x, valid_x, w)
    peaks_y = select_axis_amplitude_peaks("y", ks, amp_y, valid_y, h)
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
    axis: str, ks: np.ndarray, amplitude: np.ndarray,
    valid: np.ndarray, image_size: int,
) -> List[AxisPeriodPeak]:
    """0.73 이상인 국소 피크 중 가장 작은 k를 선택한다. 없으면 유효 최댓값 사용."""
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return []
    peak_index = int(indices[np.argmax(amplitude[indices])])
    for index in sorted(indices, key=lambda i: float(ks[i])):
        left = amplitude[index - 1] if index > 0 else -np.inf
        right = amplitude[index + 1] if index < amplitude.size - 1 else -np.inf
        if amplitude[index] >= AXIS_PEAK_RELATIVE_THRESHOLD and amplitude[index] >= max(left, right):
            peak_index = int(index)
            break
    k, value = refine_profile_peak(ks, amplitude, peak_index)
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
        k = round(peak.k, 1)
        if k <= 0:
            continue
        period = size / k
        kx, ky = (k, 0.0) if axis == "x" else (0.0, k)
        dx, dy = (period, 0.0) if axis == "x" else (0.0, period)
        candidates.append(FftPeakCandidate(
            candidate_id=len(candidates) + 1, peak_x=w // 2 + kx, peak_y=h // 2 + ky,
            kx=kx, ky=ky, dx=dx, dy=dy, period_px=period,
            angle_deg=0.0 if axis == "x" else 90.0, candidate_score=float(peak.amplitude),
        ))
    return candidates


def center_patch_box(image: Image.Image, patch_w: int, patch_h: int) -> Tuple[int, int, int, int]:
    """이미지 크기로 제한한 패치를 중앙에 배치한다. 기존 반올림 방식을 유지한다."""
    w, h = image.size
    patch_w, patch_h = max(1, min(w, patch_w)), max(1, min(h, patch_h))
    left, top = round((w - patch_w) / 2), round((h - patch_h) / 2)
    return left, top, left + patch_w, top + patch_h


def basic_ncc(first: Image.Image, second: Image.Image) -> float:
    """Basic NCC over all RGB samples, without mean subtraction."""
    if first.size != second.size:
        raise ValueError("NCC inputs must have the same size")
    a = np.asarray(first.convert("RGB"), dtype=np.float64) / 255.0
    b = np.asarray(second.convert("RGB"), dtype=np.float64) / 255.0
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.clip(np.sum(a * b) / denom, -1.0, 1.0)) if denom > 1e-12 else float("nan")


def extend_patch_edges(image_patch: Image.Image) -> Image.Image:
    """반대쪽 가장자리를 이어 붙여 가로·세로를 두 배로 확장한다."""
    pixels = np.asarray(patch.convert("RGB"))
    h, w = pixels.shape[:2]
    return Image.fromarray(np.pad(
        pixels, ((h // 2, h - h // 2), (w // 2, w - w // 2), (0, 0)), mode="wrap",
    ))


def draw_ncc_montage(
    original: Image.Image,
    patches: List[PatchResult],
    base_dx: float,
    base_dy: float,
    out_path: str,
) -> None:

    cell_w, cell_h = 630, 260
    label_h = 58
    margin = 22
    title_h = 46
    original_preview_h = 260
    section_title_h = 34
    grid_h = section_title_h + (cell_h + label_h) * 4
    canvas_w = margin * 2 + cell_w * 4
    canvas_h = margin * 3 + title_h + original_preview_h + grid_h
    montage = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(montage)
    title_font = safe_font(20)
    section_font = safe_font(18)
    label_font = safe_font(12)

    draw.text(
        (margin, margin),
        f"NCC patch ranking: dx={base_dx:.1f}px, dy={base_dy:.1f}px",
        fill=(0, 0, 0),
        font=title_font,
    )

    if original is not None:
        original_preview = original.copy().convert("RGB")
        original_preview.thumbnail((canvas_w - margin * 2, original_preview_h), Image.Resampling.LANCZOS)
        original_x = margin + (canvas_w - margin * 2 - original_preview.size[0]) // 2
        original_y = margin + title_h
        montage.paste(original_preview, (original_x, original_y))
        center_x = original_x + original_preview.size[0] // 2
        center_y = original_y + original_preview.size[1] // 2
        draw.line([(center_x, original_y), (center_x, original_y + original_preview.size[1])], fill=(255, 0, 0), width=2)
        draw.line([(original_x, center_y), (original_x + original_preview.size[0], center_y)], fill=(255, 0, 0), width=2)
        draw.rectangle(
            [original_x, original_y, original_x + original_preview.size[0], original_y + original_preview.size[1]],
            outline=(210, 210, 210),
            width=1,
        )
    draw.text((margin, margin + title_h + original_preview_h + 4), "Original image", fill=(0, 0, 0), font=label_font)

    grid_top = margin * 2 + title_h + original_preview_h
    draw.text((margin, grid_top), "NCC ranking (higher is better)", fill=(0, 0, 0), font=section_font)
    grid_top += section_title_h

    outer_edges = set()
    for position, result in enumerate(patches, start=1):
        image_patch, extended, reference = result.image_patch, result.extended, result.reference
        row_data = result.metadata
        dx_mult, dy_mult = row_data["dx_mult"], row_data["dy_mult"]
        ncc, rank = row_data["ncc_score"], row_data["rank"]
        col = (position - 1) % 4
        row = (position - 1) // 4
        cell_left = margin + col * cell_w
        cell_top = grid_top + row * (cell_h + label_h)
        panel_w = cell_w // 3
        preview_h = cell_h - 28
        for panel_index, (panel, title) in enumerate(
            ((image_patch, "Patch"), (extended, "Extended patch"), (reference, "Original crop"))
        ):
            panel_left = cell_left + panel_index * panel_w
            draw.text((panel_left + 8, cell_top + 5), title, fill=(80, 80, 80), font=label_font)
            if panel is None:
                message = "out of size"
                bounds = draw.textbbox((0, 0), message, font=section_font)
                draw.text((panel_left + (panel_w - bounds[2] + bounds[0]) // 2,
                           cell_top + cell_h // 2), message, fill=(160, 60, 60), font=section_font)
            else:
                preview = panel.copy()
                preview.thumbnail((panel_w - 16, preview_h), Image.Resampling.LANCZOS)
                montage.paste(preview, (panel_left + (panel_w - preview.width) // 2,
                                       cell_top + 22 + (preview_h - preview.height) // 2))
        right, bottom = cell_left + cell_w, cell_top + cell_h
        outer_edges.update((
            (cell_left, cell_top, right, cell_top),
            (cell_left, bottom, right, bottom),
            (cell_left, cell_top, cell_left, bottom),
            (right, cell_top, right, bottom),
        ))
        for divider in (1, 2):
            divider_x = cell_left + divider * panel_w
            draw.line([(divider_x, cell_top), (divider_x, cell_top + cell_h)], fill=(225, 225, 225), width=1)
        draw.text(
            (cell_left + 8, cell_top + cell_h + 5),
            f"Rank={rank if np.isfinite(ncc) else 'N/A'} dx x{dx_mult}, dy x{dy_mult} ({image_patch.size[0]}x{image_patch.size[1]})",
            fill=(0, 0, 0),
            font=label_font,
        )
        draw.text((cell_left + 8, cell_top + cell_h + 25), (f"NCC={ncc:.6f}" if np.isfinite(ncc) else "NCC=N/A"), fill=(0, 0, 0), font=label_font)

    # Shared edges have identical coordinates, so each thick border is drawn once.
    for edge in sorted(outer_edges):
        draw.line(edge, fill=(40, 40, 40), width=3)
    montage.save(out_path)


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
    results: List[PatchResult] = []
    for dy_mult in range(1, 5):
        for dx_mult in range(1, 5):
            patch_w = int(round(base_dx * dx_mult))
            patch_h = int(round(base_dy * dy_mult))
            left, top, right, bottom = center_patch_box(image, patch_w, patch_h)
            image_patch = image.crop((left, top, right, bottom))
            # Split odd sizes with the extra pixel on the right/bottom.
            pad_left, pad_top = image_patch.width // 2, image_patch.height // 2
            pad_right, pad_bottom = image_patch.width - pad_left, image_patch.height - pad_top
            extended = extend_patch_edges(image_patch)
            image_patch.save(os.path.join(output_dir, f"period_patch_dx{dx_mult}_dy{dy_mult}.png"))
            # Match the exact source coordinates of the extended patch.
            box = (left - pad_left, top - pad_top, right + pad_right, bottom + pad_bottom)
            fits = box[0] >= 0 and box[1] >= 0 and box[2] <= image.width and box[3] <= image.height
            reference = image.crop(box) if fits else None
            score = basic_ncc(extended, reference) if fits else float("nan")
            status = "ok" if np.isfinite(score) else ("zero_energy" if fits else "outside_image")
            metadata = dict(dx_mult=dx_mult, dy_mult=dy_mult, patch_width=image_patch.width,
                                   patch_height=image_patch.height, pad_left=pad_left, pad_top=pad_top,
                                   pad_right=pad_right, pad_bottom=pad_bottom,
                                   comparison_width=extended.width, comparison_height=extended.height,
                                   ncc_score=score, status=status, rank="")
            results.append(PatchResult(image_patch, extended, reference, metadata))

    patch_sizes = [(r.metadata["dx_mult"], r.metadata["dy_mult"], r.image_patch.width, r.image_patch.height)
                   for r in results]
    results.sort(key=lambda r: (r.reference is not None,
                 r.metadata["ncc_score"] if np.isfinite(r.metadata["ncc_score"]) else -np.inf), reverse=True)
    for rank, result in enumerate(results, start=1):
        row = result.metadata
        if row["status"] == "ok":
            row["rank"] = rank
            print(f"patch rank={rank} dx={row['dx_mult']} dy={row['dy_mult']} NCC={row['ncc_score']:.6f}")
    with open(os.path.join(output_dir, "period_patch_scores.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].metadata))
        writer.writeheader()
        writer.writerows(result.metadata for result in results)
    save_patch_box_preview(image, patch_sizes, os.path.join(output_dir, "period_patch_boxes.png"))
    draw_ncc_montage(image, results, base_dx, base_dy, os.path.join(output_dir, "period_patches_16.png"))


def main() -> None:
    # Step 14: CLI entry point. Runs the current axis-separated 1D FFT pipeline end to end.
    parser = argparse.ArgumentParser(description="Show RGB FFT center-to-peak period extraction on the full image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--fft-output-dir", default="output", help="Output directory.")
    parser.add_argument("--patch-output-dir", default="outputs/fft_patch", help="Output directory for period patches.")
    parser.add_argument("--fft-output-option", default="on", help="Choose whether to save FFT output.")
    parser.add_argument("--patch-output-option", default="on", help="Choose whether to save patch output.")
    parser.add_argument("--min-period", type=float, default=MIN_PERIOD, help="Minimum spatial period.")
    parser.add_argument("--max-period", type=float, default=MAX_PERIOD, help="Maximum spatial period.")


    args = parser.parse_args()
    if not (0 < args.min_period <= args.max_period < float("inf")):
        parser.error("Expected 0 < min-period <= max-period < infinity")

    ensure_dir(args.fft_output_dir)
    # Step 14a: Load input and build the frequency-domain visualization background.
    image, rgb = load_rgb(args.image)
    mag = fft_rgb_log_magnitude(rgb)

    fft_option = args.fft_output_option.lower()
    patch_option = args.patch_output_option.lower()

    # Step 14b: Build 1D FFT axis profiles, then hand only visualization to the draw module.
    ks, amp_x, amp_y, axis_peaks_x, axis_peaks_y = axis_period_profiles(
        rgb,
        min_period=args.min_period,
        max_period=args.max_period,
    )
    save_axis_period_functions(
        ks,
        amp_x,
        amp_y,
        axis_peaks_x,
        axis_peaks_y,
        os.path.join(args.fft_output_dir, "axis_period_functions.png"),
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
        draw_frequency_domain_peaks(
            mag,
            candidates,
            os.path.join(args.fft_output_dir, "frequency_domain_peaks.png"),
            visual_limit=visual_limit,
        )
        draw_spatial_domain_peaks(
            image,
            candidates,
            os.path.join(args.fft_output_dir, "spatial_domain_peaks.png"),
            visual_limit=visual_limit,
        )
    if patch_option == "off":
        print("Patch output is disabled. Skipping patch generation.")
    else:
        save_period_patches(image, candidates, args.patch_output_dir)

    print(f"fft_output_dir: {args.fft_output_dir}")
    print(f"image_size: width={image.size[0]}, height={image.size[1]}")

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
