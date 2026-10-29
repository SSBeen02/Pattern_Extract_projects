"""FFT3의 후보를 원본 중앙 패치 NCC와 평균 패치 VGG19 점수로 평가한다.

1. FFT3의 RGB FFT와 축별 다수결을 사용하되, FFT6의 peak 임계값은
   최대진폭 * max(0.3, 1 - 0.3 * (k - 1) / (k_max - 1))로 정한다.
   선택한 주기로 1~3배 후보를 만든다.
2. 각 후보는 중앙 및 상하좌우 5영역을 평균한 패치로 만든다.
3. NCC는 평균 전 원본 중앙 패치를 한 주기(dx, dy)만큼 반복 확장하고,
   확장 패치의 바깥 여백과 원본의 같은 위치를
   비교한다. 비교 영역이 원본 밖이면 NCC는 미판정이다.
4. conv4_4 점수는 이 평균 패치를 원본 크기로 타일링하고 VGG19의
   conv4_4 특징맵 전체 공간 위치의 채널 벡터 cosine 평균으로 비교한다.
5. 최종 점수 = 0.9 * NCC + 0.1 * conv4_4. 두 값이 모두 유효할 때만 순위를
   부여한다. 후보별 원점수, 결합점수, 상태를 CSV와 montage에 기록한다.

FFT5의 코드는 수정하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

import period_candidates_fft3 as fft3
from period_output_naming import candidate_prefix, image_id, save_compact_montage

THRESHOLD_S = 0.3
THRESHOLD_MIN = 0.3


def select_axis_amplitude_peaks(axis, ks, amplitude, valid, image_size):
    """FFT6-only peak selection; switch the two threshold lines to revert."""
    indices = np.flatnonzero(valid & np.isfinite(ks) & (ks > 0))
    if indices.size == 0:
        return []
    finite_indices = indices[np.isfinite(amplitude[indices])]
    if not finite_indices.size or not np.any(amplitude[finite_indices] > 0):
        return []
    max_amplitude = float(np.max(amplitude[finite_indices]))
    k_max = float(ks[-1])
    peak_indices = []
    for index in indices:
        k = float(ks[index])
        # To restore FFT3's old rule, comment the next line and uncomment the one below.
        threshold = max_amplitude * max(THRESHOLD_MIN, 1 - THRESHOLD_S * (k - 1) / max(1, k_max - 1))
        # threshold = max_amplitude / (k ** fft3.alpha)
        left = amplitude[index - 1] if index > 0 else -np.inf
        right = amplitude[index + 1] if index < amplitude.size - 1 else -np.inf
        if np.isfinite(amplitude[index]) and amplitude[index] >= threshold and amplitude[index] >= max(left, right):
            peak_indices.append(index)
    final_index = max(peak_indices, key=lambda i: float(ks[i]), default=-1)
    if final_index < 0:
        return []
    k, value = fft3.refine_profile_peak(ks, amplitude, final_index)
    return [fft3.AxisPeriodPeak(axis, k, image_size / k if k > 0 else float("inf"), value, "amplitude")]


def ncc_for_patch(image, result):
    """Compare the raw center crop's repeated margin with the original margin."""
    if result.reference is None:
        return float("nan")
    row = result.metadata
    patch_w, patch_h = row["patch_width"], row["patch_height"]
    left = round((image.width - patch_w) / 2)
    top = round((image.height - patch_h) / 2)
    patch = image.crop((left, top, left + patch_w, top + patch_h))
    extra_w = result.extended.width - patch_w
    extra_h = result.extended.height - patch_h
    extended = fft3.extend_patch_edges(patch, extra_w, extra_h)
    mask = np.ones((extended.height, extended.width), dtype=bool)
    mask[extra_h // 2:extra_h // 2 + patch.height,
         extra_w // 2:extra_w // 2 + patch.width] = False
    return fft3.opencv_ncc(extended, result.reference, mask)


def combined_patches(image, candidates):
    """Keep FFT3 patch/tile generation, then add its NCC score to conv4_4 score."""
    conv_features = fft3.extract_vgg19_features(image, "conv")
    results = fft3.create_period_patches(
        image, candidates, score_method="vgg19", feature_maps=conv_features,
        vgg_feature="conv",
    )
    for result in results:
        row = result.metadata
        conv = row["vgg19_score"]
        ncc = ncc_for_patch(image, result)
        combined = 0.9 * ncc + 0.1 * conv
        row.update(conv4_4_score=conv, ncc_score=ncc, combined_score=combined,
                   score_method="combined", score_metric="half_ncc_half_vgg19_conv4_4",
                   status="ok" if math.isfinite(combined) else
                   ("outside_image" if result.reference is None else "invalid_score"))
    return fft3.rank_period_patches(results, "combined")


def save_montage(image, result, output_path):
    """Keep the FFT6 entry point while sharing its three-column layout."""
    save_compact_montage(image, result, output_path)


def save_results(image, results, output_dir, image_path, save_patches=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{candidate_prefix(image_path, results[0].metadata)}.csv"
    fields = ["rank", "dx_mult", "dy_mult", "patch_width", "patch_height",
              "ncc_score", "conv4_4_score", "combined_score", "status"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = result.metadata
            writer.writerow({field: row[field] for field in fields})
    if save_patches:
        for result in results:
            row = result.metadata
            prefix = candidate_prefix(image_path, row)
            if row["dx_mult"] == row["dy_mult"] == 1:
                result.image_patch.save(output_dir / f"{image_id(image_path)}_original_patch.png")
            filename = f"{prefix}.png"
            save_montage(image, result, output_dir / filename)
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Rank FFT3 averaged patches by 50% NCC and 50% VGG19 conv4_4."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", "--output-dir", default="outputs/fft6")
    parser.add_argument("--fout-opt", choices=("on", "off"), default="on")
    parser.add_argument("--pout-opt", choices=("on", "off"), default="on")
    args = parser.parse_args()

    image, rgb = fft3.load_rgb(args.image)
    ks, _, _, _, _, vote_x, vote_y = fft3.axis_period_profiles(
        rgb, include_votes=True, peak_selector=select_axis_amplitude_peaks)
    candidates = fft3.candidates_from_axis_period_peaks(
        rgb.shape[:2], vote_x.peaks, vote_y.peaks)
    if not candidates:
        raise SystemExit("No patch candidates: at least one FFT axis has no selected peak.")
    results = combined_patches(image, candidates)
    fft3.ensure_dir(args.output_dir)
    fft3.save_axis_period_functions(
        ks, vote_x.amplitude, vote_y.amplitude, vote_x.peaks, vote_y.peaks,
        os.path.join(args.output_dir, f"{image_id(args.image)}_axis.png"),
        os.path.basename(args.image),
    )
    if args.fout_opt == "on":
        fft3.draw_spatial_domain_peaks(
            image, candidates, os.path.join(args.output_dir, f"{image_id(args.image)}_period.png"),
            visual_limit=2,
        )
    csv_path = save_results(image, results, args.output_dir, args.image, args.pout_opt == "on")
    print(f"CSV: {csv_path}")
    for result in results:
        row = result.metadata
        print(f"rank={row['rank'] or 'N/A'} dx={row['dx_mult']} dy={row['dy_mult']} "
              f"ncc={row['ncc_score']:.6f} conv4_4={row['conv4_4_score']:.6f} "
              f"combined={row['combined_score']:.6f} status={row['status']}")


if __name__ == "__main__":
    main()
