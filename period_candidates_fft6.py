"""FFT3의 5영역 평균 패치를 NCC와 VGG19 conv4_4 점수로 함께 평가한다.

1. FFT3의 RGB FFT, 축별 다수결, 1~3배 후보 생성 방식을 사용한다.
2. 각 후보는 중앙 및 상하좌우 5영역을 평균한 패치로 만든다.
3. NCC는 FFT3과 동일하게 확장 패치의 바깥 여백과 원본의 같은 위치를
   비교한다. 비교 영역이 원본 밖이면 NCC는 미판정이다.
4. conv4_4 점수는 이 평균 패치를 원본 크기로 타일링하고 VGG19의
   conv4_4 특징맵 전체 공간 위치의 채널 벡터 cosine 평균으로 비교한다.
5. 최종 점수 = 0.5 * NCC + 0.5 * conv4_4. 두 값이 모두 유효할 때만 순위를
   부여한다. 후보별 원점수, 결합점수, 상태를 CSV와 montage에 기록한다.

FFT3과 FFT5의 코드는 수정하지 않는다.
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


def ncc_for_patch(result):
    """Reproduce FFT3's margin-only NCC for an already averaged patch."""
    if result.reference is None:
        return float("nan")
    patch = result.image_patch
    extended = result.extended
    extra_w = extended.width - patch.width
    extra_h = extended.height - patch.height
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
        ncc = ncc_for_patch(result)
        combined = 0.5 * ncc + 0.5 * conv
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
    ks, _, _, _, _, vote_x, vote_y = fft3.axis_period_profiles(rgb, include_votes=True)
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
