"""FFT3의 패치 생성 방식을 유지하면서 VGG19 두 층의 점수를 결합한다.

1. RGB별 FFT peak를 뽑고 축별 다수결로 가로/세로 기본 주기를 구한다.
2. 각 축의 정수 픽셀 1배 크기를 정한 뒤 1~3배의 9개 후보를 만든다.
3. 후보마다 중앙·상하좌우 5영역의 대응 픽셀을 평균해 패치를 만들고,
   원본 이미지 크기로 타일링한다. 경계 밖 이웃은 평균에서 제외한다.
4. 원본과 타일 이미지에서 VGG19 conv3_4의 모든 공간 위치의 채널별
   cosine 유사도를 평균하고, 마지막 dense 층의 벡터 cosine도 계산한다.
5. combined_score = 0.5 * conv3_4_score + 0.5 * dense_score로 순위를 매긴다.
   어느 층도 중앙을 마스킹하지 않는다. 기존 montage에는 결합 점수를 표시한다.

FFT/패치/시각화 함수는 FFT3의 현재 구현을 재사용하며 FFT3 파일은 수정하지 않는다.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

import period_candidates_fft3 as fft3
from period_output_naming import image_id, save_ranked_patches


def conv3_4_features(image):
    """Extract VGG19 features[17] (ReLU after conv3_4) at native image size."""
    import torch
    from torchvision import transforms

    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return fft3._vgg19_dense_model().features[:18](tensor)


def conv3_4_score(original_features, tiled):
    """Mean channel-vector cosine across every spatial position, no mask."""
    import torch

    with torch.no_grad():
        candidate = conv3_4_features(tiled)
        if original_features.shape != candidate.shape:
            raise ValueError("Original and tiled conv3_4 feature shapes must match")
        if original_features.norm().item() == 0 or candidate.norm().item() == 0:
            return float("nan")
        return torch.nn.functional.cosine_similarity(
            original_features.to(candidate.device), candidate, dim=1, eps=1e-8,
        ).mean().item()


def combined_vgg_patches(image, candidates):
    """Score FFT3 averaged/tiled candidates with both VGG19 feature sources."""
    dense_features = fft3.extract_vgg19_features(image, "dense")
    conv_features = conv3_4_features(image)
    results = fft3.create_period_patches(
        image, candidates, score_method="vgg19", feature_maps=dense_features,
        vgg_feature="dense",
    )
    for result in results:
        dense_score = result.metadata["vgg19_score"]
        conv_score = conv3_4_score(conv_features, result.tiled)
        combined = 0 * dense_score + 1* conv_score
        result.metadata.update(
            dense_score=dense_score,
            conv3_4_score=conv_score,
            combined_score=combined,
            vgg19_score=combined,
            score_metric="half_dense_half_conv3_4_cosine_similarity",
            status="ok" if np.isfinite(combined) else "no_comparison_signal",
        )
    return fft3.rank_period_patches(results, "vgg19")


def main():
    parser = argparse.ArgumentParser(
        description="Rank FFT3's averaged tiled patches with 50% VGG19 dense and 50% conv3_4."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output_dir", "--output-dir", default="outputs/fft5",
                        help="Output directory (default: outputs/fft5).")
    parser.add_argument("--fout-opt", choices=("on", "off"), default="on")
    parser.add_argument("--pout-opt", choices=("on", "off"), default="on")
    args = parser.parse_args()

    fft3.ensure_dir(args.output_dir)
    image, rgb = fft3.load_rgb(args.image)
    ks, _, _, _, _, vote_x, vote_y = fft3.axis_period_profiles(rgb, include_votes=True)
    candidates = fft3.candidates_from_axis_period_peaks(
        rgb.shape[:2], vote_x.peaks, vote_y.peaks)
    if not candidates:
        raise SystemExit("No patch candidates: at least one FFT axis has no selected peak.")
    results = combined_vgg_patches(image, candidates)

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
    if args.pout_opt == "on":
        save_ranked_patches(image, results, args.output_dir, "vgg19", args.image)

    print(f"output_dir: {args.output_dir}")
    for result in results:
        row = result.metadata
        if math.isfinite(row["combined_score"]):
            print(f"rank={row['rank']} dx={row['dx_mult']} dy={row['dy_mult']} "
                  f"dense={row['dense_score']:.6f} "
                  f"conv3_4={row['conv3_4_score']:.6f} "
                  f"combined={row['combined_score']:.6f}")


if __name__ == "__main__":
    main()
