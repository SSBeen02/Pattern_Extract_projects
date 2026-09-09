"""
Seamless Fabric Tiling Pipeline #4
=================================

이 버전은 "심리스(seamless)로 반복 가능한 최소 패치"를 찾은 뒤,
선택된 패치를 변형 없이 단순 반복해 최종 이미지를 만드는 데 집중합니다.
즉, 경계 블렌딩/합성으로 이음새를 숨기기보다,
초기에 "잘 반복되는 패치"를 고르는 전략입니다.

핵심 처리 흐름:
     1. TexTile 점수가 최대가 되는 회전 각도 theta를 찾아 입력을 정렬합니다.
         (argmax TexTile(Rotate(I, theta)))
     2. 정렬된 이미지에서 VGG 특성 기반 주기 후보를 추정합니다.
     3. FFT 전력 스펙트럼 기반 주기 후보를 추정합니다. (옵션)
     4. 중심 크롭 기준 TexTile 점수로 반복 패턴 크기 후보를 추가합니다.
         (argmax TexTile(Crop(I, (h, w))))
     5. 각 후보 크기에 대해 오프셋을 스캔하고,
         style loss + RTV loss + TexTile loss + area penalty의
         결합 점수로 최종 패치를 선택합니다.
     6. 모든 후보 패치를 저장하고,
         최종 패치를 주기적으로 반복 타일링해 출력 이미지를 생성합니다.
"""
import argparse
import math
import os
import re
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

try:
    import torchvision.models as tvm

    _HAS_TV = True
except Exception:
    _HAS_TV = False


VGG_LAYER = 20
VGG_LAYER_STRIDE = 8
TEXTILE_LOGIT_SCALE = 0.25


# =====================================================================
# 입출력(IO) 및 기본 유틸
# =====================================================================
def load_image(path, device, max_side=None):
    img = Image.open(path).convert("RGB")
    if max_side is not None and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.LANCZOS,
        )
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def save_image(tensor, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    arr = (
        tensor.detach()
        .squeeze(0)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def parse_size(value):
    height, width = value.lower().split("x")
    return int(height), int(width)


def default_textile_checkpoint():
    for path in ("textile.pth", os.path.join("textile", "models", "textile.pth")):
        if os.path.exists(path):
            return path
    return "textile.pth"


def center_crop_tensor(img, crop_h, crop_w):
    img_h, img_w = img.shape[-2:]
    crop_h = max(1, min(int(crop_h), img_h))
    crop_w = max(1, min(int(crop_w), img_w))
    oy = max(0, (img_h - crop_h) // 2)
    ox = max(0, (img_w - crop_w) // 2)
    crop = img[:, :, oy : oy + crop_h, ox : ox + crop_w].contiguous()
    return crop, (oy, ox, crop_h, crop_w)


def _valid_rotation_crop_hw(img_h, img_w, angle_rad):
    c = abs(math.cos(angle_rad))
    s = abs(math.sin(angle_rad))
    denom_w = img_w * c + img_h * s
    denom_h = img_w * s + img_h * c
    scale = min(
        img_w / max(denom_w, 1e-8),
        img_h / max(denom_h, 1e-8),
        1.0,
    )
    crop_h = max(1, min(img_h, int(math.floor(img_h * scale)) - 2))
    crop_w = max(1, min(img_w, int(math.floor(img_w * scale)) - 2))
    return crop_h, crop_w


def rotate_image(img, angle_degrees, crop_valid=True):
    angle_degrees = float(angle_degrees) % 360.0
    if abs(angle_degrees) < 1e-8:
        img_h, img_w = img.shape[-2:]
        return img.contiguous(), (0, 0, img_h, img_w)

    batch, _, img_h, img_w = img.shape
    angle_rad = math.radians(angle_degrees)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    # grid_sample uses normalized coordinates. The h/w compensation keeps the
    # rotation geometrically correct for non-square images.
    theta = img.new_tensor(
        [
            [
                [c, s * img_h / img_w, 0.0],
                [-s * img_w / img_h, c, 0.0],
            ]
        ]
    ).repeat(batch, 1, 1)
    grid = F.affine_grid(theta, img.size(), align_corners=False)
    rotated = F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    if not crop_valid:
        return rotated.contiguous(), (0, 0, img_h, img_w)

    crop_h, crop_w = _valid_rotation_crop_hw(img_h, img_w, angle_rad)
    return center_crop_tensor(rotated, crop_h, crop_w)


# =====================================================================
# TexTile 모델 래퍼
# =====================================================================
class TextileTileabilityScorer(torch.nn.Module):
    """TexTile 추론식을 감싼 타일 가능성 점수 계산기.

    입력 이미지를 (필요 시) 반복/리사이즈/정규화한 뒤 모델에 넣고,
    로짓 평균을 시그모이드로 변환해 "타일 가능성" 점수를 반환합니다.
    """

    def __init__(
        self,
        model_path="textile.pth",
        resolution=(512, 512),
        number_tiles=2,
        logit_scale=TEXTILE_LOGIT_SCALE,
        device=None,
    ):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TexTile checkpoint not found: {model_path}")

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.resolution = tuple(resolution)
        self.number_tiles = int(number_tiles)
        self.logit_scale = float(logit_scale)

        self.model = self._load_model(model_path)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        elif self.device.type == "cuda" and hasattr(self.model, "cuda"):
            self.model = self.model.cuda()
        self.model.eval()
        if hasattr(self.model, "parameters"):
            for param in self.model.parameters():
                param.requires_grad_(False)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.to(self.device)

    def _load_model(self, model_path):
        try:
            from textile.utils.create_model import CreateModel

            return CreateModel(model_path)
        except Exception as package_exc:
            try:
                return torch.jit.load(model_path, map_location=self.device)
            except Exception as jit_exc:
                message = (
                    "Could not load the TexTile model. Install the original "
                    "TexTile package that provides textile.utils.create_model, "
                    "or pass a TorchScript TexTile checkpoint. "
                    f"Package loader error: {package_exc}. "
                    f"TorchScript loader error: {jit_exc}."
                )
                raise ImportError(message) from package_exc

    @staticmethod
    def _unwrap_model_output(result):
        if isinstance(result, dict):
            if "logits" in result:
                return result["logits"]
            if "logit" in result:
                return result["logit"]
            return next(iter(result.values()))
        if isinstance(result, (tuple, list)):
            return result[0]
        return result

    @torch.no_grad()
    def forward(
        self,
        image,
        return_logits=True,
        normalize=True,
        rescale=True,
        tile=True,
    ):
        assert image.dim() == 4
        assert image.size(1) == 3

        x = image.to(self.device)
        if tile:
            x = x.repeat(1, 1, self.number_tiles, self.number_tiles)
        if rescale:
            x = F.interpolate(
                x,
                size=self.resolution,
                mode="bilinear",
                align_corners=False,
            )
        if normalize:
            x = (x - self.mean) / self.std

        logits = self._unwrap_model_output(self.model(x.float()))
        if return_logits:
            return logits
        return torch.sigmoid(self.logit_scale * logits)

    @torch.no_grad()
    def logit(self, image):
        logit, _ = self.logit_and_tileability(image)
        return logit

    @torch.no_grad()
    def tileability(self, image):
        _, score = self.logit_and_tileability(image)
        return score

    @torch.no_grad()
    def logit_and_tileability(self, image):
        logits = self.forward(image, return_logits=True)
        mean_logit = logits.mean()
        score = torch.sigmoid(self.logit_scale * mean_logit)
        return (
            float(mean_logit.detach().item()),
            float(score.detach().item()),
        )


# =====================================================================
# TexTile 기반 회전 탐색 및 크롭 크기 탐색
# =====================================================================
@torch.no_grad()
def find_best_rotation(
    img,
    textile_scorer,
    coarse_step=5.0,
    refine_step=1.0,
    crop_valid=True,
):
    """TexTile 점수가 최대가 되는 회전 각도를 탐색합니다.

    1) coarse_step 간격으로 0~360도 전역 탐색
    2) 최고 각도 주변을 refine_step으로 국소 정밀 탐색
    3) 최적 회전 이미지와 상위 평가 결과를 반환
    """
    if textile_scorer is None:
        img_h, img_w = img.shape[-2:]
        return img, {
            "angle": 0.0,
            "score": None,
            "crop": (0, 0, img_h, img_w),
            "evaluations": [],
        }

    coarse_step = max(float(coarse_step), 1e-6)
    refine_step = max(float(refine_step), 1e-6)

    coarse_angles = np.arange(0.0, 360.0, coarse_step, dtype=np.float32)
    evaluations = []

    def evaluate(angle):
        rotated, crop = rotate_image(img, angle, crop_valid=crop_valid)
        score = textile_scorer.tileability(rotated)
        return {
            "angle": float(angle % 360.0),
            "score": score,
            "crop": crop,
        }

    for angle in coarse_angles:
        evaluations.append(evaluate(float(angle)))

    coarse_best = max(evaluations, key=lambda item: item["score"])
    refine_start = coarse_best["angle"] - coarse_step
    refine_end = coarse_best["angle"] + coarse_step
    refine_angles = np.arange(
        refine_start,
        refine_end + refine_step * 0.5,
        refine_step,
        dtype=np.float32,
    )
    seen = {round(item["angle"], 6) for item in evaluations}
    for angle in refine_angles:
        key = round(float(angle % 360.0), 6)
        if key in seen:
            continue
        seen.add(key)
        evaluations.append(evaluate(float(angle)))

    best = max(evaluations, key=lambda item: item["score"])
    rotated, crop = rotate_image(img, best["angle"], crop_valid=crop_valid)
    best = dict(best)
    best["crop"] = crop
    best["evaluations"] = sorted(
        evaluations,
        key=lambda item: item["score"],
        reverse=True,
    )[:10]
    return rotated, best


def _sample_axis_sizes(length, samples):
    samples = max(1, int(samples))
    if length <= samples:
        return list(range(1, length + 1))
    values = np.linspace(1, length, samples)
    return sorted({max(1, min(length, int(round(value)))) for value in values})


@torch.no_grad()
def detect_textile_pattern_sizes(
    img,
    textile_scorer,
    size_grid=16,
    top_k=6,
    refine_top_k=4,
):
    """TexTile 점수가 높은 크롭 크기 후보를 찾습니다.

    중심 크롭 기준으로 (h, w) 격자를 샘플링해 coarse 점수를 구하고,
    상위 후보 주변을 더 촘촘히 재탐색해 최종 top-k 크기를 반환합니다.
    """
    if textile_scorer is None:
        return []

    img_h, img_w = img.shape[-2:]
    h_values = _sample_axis_sizes(img_h, size_grid)
    w_values = _sample_axis_sizes(img_w, size_grid)

    def score_size(size_hw):
        patch, crop = center_crop_tensor(img, size_hw[0], size_hw[1])
        score = textile_scorer.tileability(patch)
        return {
            "source": "textile-size",
            "size": (int(size_hw[0]), int(size_hw[1])),
            "period": (int(size_hw[0]), int(size_hw[1])),
            "center_crop": crop,
            "textile_size_score": score,
        }

    coarse_sizes = {(h, w) for h in h_values for w in w_values}
    coarse_results = [score_size(size) for size in sorted(coarse_sizes)]
    coarse_results.sort(key=lambda item: item["textile_size_score"], reverse=True)

    h_step = max(1, int(round(img_h / max(size_grid - 1, 1))))
    w_step = max(1, int(round(img_w / max(size_grid - 1, 1))))
    h_refine_step = max(1, h_step // 4)
    w_refine_step = max(1, w_step // 4)

    refine_sizes = set(coarse_sizes)
    for item in coarse_results[: max(1, int(refine_top_k))]:
        h0, w0 = item["size"]
        h_start = max(1, h0 - h_step)
        h_end = min(img_h, h0 + h_step)
        w_start = max(1, w0 - w_step)
        w_end = min(img_w, w0 + w_step)
        for h in range(h_start, h_end + 1, h_refine_step):
            for w in range(w_start, w_end + 1, w_refine_step):
                refine_sizes.add((h, w))

    scored = {item["size"]: item for item in coarse_results}
    for size in sorted(refine_sizes - set(scored)):
        scored[size] = score_size(size)

    results = list(scored.values())
    results.sort(key=lambda item: item["textile_size_score"], reverse=True)
    return results[: max(1, int(top_k))]


# =====================================================================
# VGG/FFT 기반 주기(period) 후보 추정
# =====================================================================
def _first_peak(profile, lo, hi):
    lo = max(1, int(lo))
    hi = min(int(hi), profile.numel() - 1)
    if hi <= lo:
        return lo
    segment = profile[lo : hi + 1]
    return int(torch.argmax(segment).item()) + lo


@torch.no_grad()
def estimate_period_fft_power(img, min_period=1, max_period=None):
    """FFT 전력 스펙트럼 자기상관 피크로 반복 주기를 추정합니다."""
    gray = img.mean(1, keepdim=True)
    centered = gray - gray.mean()
    img_h, img_w = centered.shape[-2:]
    fft = torch.fft.rfft2(centered)
    power = fft * torch.conj(fft)
    ac = torch.fft.irfft2(power, s=(img_h, img_w)).real[0, 0]
    ac = ac / (ac.abs().max() + 1e-8)

    max_p_h = max_period or img_h
    max_p_w = max_period or img_w
    py = _first_peak(ac[:, 0], min_period, max_p_h)
    px = _first_peak(ac[0, :], min_period, max_p_w)

    # 너무 작은 주기를 방지하기 위한 하한.
    # 단, 이미지가 더 작다면 해당 축의 최대 크기로 제한합니다.
    MIN_SIZE = 100
    py = int(max(1, min(max(py, MIN_SIZE), img_h)))
    px = int(max(1, min(max(px, MIN_SIZE), img_w)))
    return py, px


class VGGActivations(torch.nn.Module):
    """고정된 VGG19의 20번 레이어 특성 맵을 반환합니다."""

    def __init__(self, device):
        super().__init__()
        assert _HAS_TV, "torchvision is required for VGG extraction"
        try:
            vgg = tvm.vgg19(weights=tvm.VGG19_Weights.DEFAULT).features
        except Exception as exc:
            print(
                "[warn] Failed to load pretrained VGG19 weights "
                f"({exc}); using randomly initialized VGG19."
            )
            vgg = tvm.vgg19(weights=None).features

        self.vgg = vgg.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.layer_idx = VGG_LAYER
        self.stride = VGG_LAYER_STRIDE
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.to(device)

    @torch.no_grad()
    def forward(self, img):
        x = (img - self.mean) / self.std
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx == self.layer_idx:
                return x
        return x


def _period_from_feature(
    feat,
    stride,
    min_period=1,
    max_period=None,
    top_k=96,
    peak_k=5,
    axis_tol=2,
):
    """활성 피크 간 거리 투표(Hough 유사)로 반복 주기를 추정합니다.

    채널별 강한 활성 위치를 뽑고,
    수평/수직 축 방향 거리 분포(histogram)를 누적해
    지배적인 반복 간격을 주기로 선택합니다.
    """
    channels, h, w = feat.shape
    device = feat.device
    max_period = max_period or max(h, w) * stride

    channel_score = feat.view(channels, -1).amax(1)
    keep = torch.topk(channel_score, min(top_k, channels)).indices

    hist_len = max(h, w) + 1
    hx = torch.zeros(hist_len, device=device)
    hy = torch.zeros(hist_len, device=device)

    for channel in keep.tolist():
        activation = feat[channel]
        pooled = F.max_pool2d(
            activation[None, None],
            peak_k,
            1,
            peak_k // 2,
        )[0, 0]
        threshold = activation.mean() + activation.std()
        peaks = torch.nonzero(
            (activation >= pooled) & (activation > threshold),
            as_tuple=False,
        )
        peak_count = peaks.shape[0]
        if peak_count < 2 or peak_count > 400:
            continue

        ys, xs = peaks[:, 0], peaks[:, 1]
        dy = (ys[:, None] - ys[None, :]).abs()
        dx = (xs[:, None] - xs[None, :]).abs()
        horizontal_mask = (dy <= axis_tol) & (dx > 0)
        vertical_mask = (dx <= axis_tol) & (dy > 0)

        if horizontal_mask.any():
            hx += torch.bincount(
                dx[horizontal_mask].clamp(max=hist_len - 1),
                minlength=hist_len,
            ).float()
        if vertical_mask.any():
            hy += torch.bincount(
                dy[vertical_mask].clamp(max=hist_len - 1),
                minlength=hist_len,
            ).float()

    def dominant(hist, default_period):
        lo = max(1, math.ceil(min_period / stride))
        hi = min(hist.numel() - 1, max(1, int(max_period // stride)))
        if hi <= lo:
            return default_period
        segment = hist[lo : hi + 1]
        if float(segment.max()) <= 0:
            return default_period
        return (int(torch.argmax(segment).item()) + lo) * stride

    py = dominant(hy, stride)
    px = dominant(hx, stride)
    return py, px, hy, hx


def build_patch_candidates(periods, image_hw, textile_size_candidates):
    """주기/크기 후보를 중복 없이 통합합니다.

    동일한 (h, w) 크기는 하나로 합치고,
    어떤 소스(vgg/fft/textile-size)에서 왔는지 source에 누적 기록합니다.
    """
    img_h, img_w = image_hw
    candidates = []
    by_size = {}

    def add_candidate(source, size, period=None, textile_size_score=None):
        patch_h = max(1, min(img_h, int(size[0])))
        patch_w = max(1, min(img_w, int(size[1])))
        key = (patch_h, patch_w)
        if key in by_size:
            existing = by_size[key]
            if source not in existing["source"].split("+"):
                existing["source"] = f"{existing['source']}+{source}"
            if textile_size_score is not None:
                existing["textile_size_score"] = textile_size_score
            return

        candidate = {
            "source": source,
            "size": key,
            "period": tuple(period or key),
            "search_hw": tuple(period or key),
            "textile_size_score": textile_size_score,
        }
        by_size[key] = candidate
        candidates.append(candidate)

    for source, period in periods:
        add_candidate(source, period, period=period)

    for item in textile_size_candidates:
        add_candidate(
            item["source"],
            item["size"],
            period=item["period"],
            textile_size_score=item.get("textile_size_score"),
        )

    return candidates


# =====================================================================
# 패치 점수 계산
# =====================================================================
def patchwise_tv(img, p=4):
    h, w = img.shape[-2:]
    zero = img.new_zeros(())
    if w > 1:
        px = min(int(p), w - 1)
        dx = (img[..., :, px:] - img[..., :, :-px]).abs().mean()
    else:
        dx = zero
    if h > 1:
        py = min(int(p), h - 1)
        dy = (img[..., py:, :] - img[..., :-py, :]).abs().mean()
    else:
        dy = zero
    return dx, dy


def rtv(img, p=4, eps=1e-8):
    """Relative Total Variation 계산.

    값이 낮을수록 이동/반복에 따른 경계 불연속이 작아
    타일링 친화적이라고 해석합니다.
    """
    h, w = img.shape[-2:]
    shifted = torch.roll(img, shifts=(h // 2, w // 2), dims=(2, 3))
    dx0, dy0 = patchwise_tv(img, p)
    dx1, dy1 = patchwise_tv(shifted, p)
    rx = (dx1 - dx0).abs() / (dx0 + eps)
    ry = (dy1 - dy0).abs() / (dy0 + eps)
    return 0.5 * (rx + ry)


class VGGStyleLayer20(torch.nn.Module):
    """VGG19 20번 레이어 Gram 행렬 기반 스타일 손실 계산기."""

    def __init__(self, device, resolution=(256, 256)):
        super().__init__()
        assert _HAS_TV, "torchvision is required for VGG style loss"
        try:
            vgg = tvm.vgg19(weights=tvm.VGG19_Weights.DEFAULT).features
        except Exception as exc:
            print(
                "[warn] Failed to load pretrained VGG19 style weights "
                f"({exc}); using randomly initialized VGG19."
            )
            vgg = tvm.vgg19(weights=None).features

        self.vgg = vgg.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.layer_idx = VGG_LAYER
        self.resolution = tuple(resolution)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.to(device)

    def _feat(self, img):
        x = F.interpolate(
            img,
            size=self.resolution,
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx == self.layer_idx:
                return x
        return x

    @staticmethod
    def gram(feat):
        batch, channels, h, w = feat.shape
        flat = feat.view(batch, channels, h * w)
        return flat @ flat.transpose(1, 2) / (channels * h * w)

    @torch.no_grad()
    def image_gram(self, img):
        return self.gram(self._feat(img))

    @torch.no_grad()
    def loss_to_gram(self, img, target_gram):
        return F.mse_loss(self.image_gram(img), target_gram)

    def forward(self, out_img, target_img):
        return F.mse_loss(self.image_gram(out_img), self.image_gram(target_img))


class TileabilityLoss:
    """패치 종합 점수 계산기.

    총점 = style + RTV + TexTile + area penalty
    점수가 낮을수록 최종 패치 후보로 유리합니다.
    """

    def __init__(
        self,
        p=4,
        textile_scorer=None,
        style_fn=None,
        style_target_gram=None,
        rtv_weight=1.0,
        textile_weight=1.0,
        style_weight=1.0,
        area_weight=1e-4,
    ):
        self.p = p
        self.textile_scorer = textile_scorer
        self.style_fn = style_fn
        self.style_target_gram = style_target_gram
        self.rtv_weight = float(rtv_weight)
        self.textile_weight = float(textile_weight)
        self.style_weight = float(style_weight)
        self.area_weight = float(area_weight)

    @torch.no_grad()
    def score_parts(self, img):
        rtv_loss = float(rtv(img, self.p).detach().item())

        textile_score = None
        textile_loss = 0.0
        textile_logit = None
        if self.textile_scorer is not None:
            textile_logit, textile_score = (
                self.textile_scorer.logit_and_tileability(img)
            )
            textile_loss = 1.0 - textile_score

        style_loss = 0.0
        if self.style_fn is not None and self.style_target_gram is not None:
            style_value = self.style_fn.loss_to_gram(img, self.style_target_gram)
            style_loss = float(style_value.detach().item())

        patch_h, patch_w = img.shape[-2:]
        area_loss = math.log(max(1, int(patch_h) * int(patch_w)))
        total = (
            self.style_weight * style_loss
            + self.rtv_weight * rtv_loss
            + self.textile_weight * textile_loss
            + self.area_weight * area_loss
        )
        return {
            "total": total,
            "style_loss": style_loss,
            "rtv_loss": rtv_loss,
            "textile_loss": textile_loss,
            "textile_score": textile_score,
            "textile_logit": textile_logit,
            "area_loss": area_loss,
            "area": int(patch_h) * int(patch_w),
        }

    @torch.no_grad()
    def score(self, img):
        return self.score_parts(img)["total"]


@torch.no_grad()
def select_patch_at_size(img, patch_hw, patch_scorer, offset_step=8, search_hw=None):
    """고정된 패치 크기에서 최적 오프셋(y, x)을 선택합니다.

    지정된 탐색 범위 안에서 offset_step 격자로 스캔하며,
    종합 점수가 최소인 위치를 반환합니다.
    """
    img_h, img_w = img.shape[-2:]
    patch_h = max(1, min(int(patch_hw[0]), img_h))
    patch_w = max(1, min(int(patch_hw[1]), img_w))
    search_h, search_w = search_hw or patch_hw

    oy_stop = max(1, min(int(search_h), img_h - patch_h) + 1)
    ox_stop = max(1, min(int(search_w), img_w - patch_w) + 1)
    step = max(1, int(offset_step))

    best = None
    for oy in range(0, oy_stop, step):
        for ox in range(0, ox_stop, step):
            patch = img[:, :, oy : oy + patch_h, ox : ox + patch_w]
            parts = patch_scorer.score_parts(patch)
            score = parts["total"]
            if best is None or score < best[0]:
                best = (score, oy, ox, parts)

    _, oy, ox, parts = best
    patch = img[:, :, oy : oy + patch_h, ox : ox + patch_w].contiguous()
    return patch, (oy, ox, patch_h, patch_w), parts


@torch.no_grad()
def extract_tile_patch(
    img,
    act_fn,
    patch_scorer,
    textile_scorer=None,
    offset_step=8,
    use_fft=True,
    textile_size_grid=16,
    textile_size_top_k=6,
):
    """VGG/FFT 주기 후보와 TexTile 크기 후보를 결합해 최적 패치를 추출합니다."""
    img_h, img_w = img.shape[-2:]

    # Step 1) 특성 기반 구조(VGG)와 주파수 기반 구조(FFT)에서 주기 후보를 수집합니다.
    feat = act_fn(img)[0]
    py_vgg, px_vgg, _, _ = _period_from_feature(feat, act_fn.stride)

    periods = [("vgg20", (py_vgg, px_vgg))]
    py_fft = px_fft = None
    if use_fft:
        py_fft, px_fft = estimate_period_fft_power(img)
        periods.append(("fft", (py_fft, px_fft)))

    # Step 2) TexTile 점수 기반 크롭 크기 후보를 추가합니다.
    textile_sizes = detect_textile_pattern_sizes(
        img,
        textile_scorer,
        size_grid=textile_size_grid,
        top_k=textile_size_top_k,
    )

    # Step 3) 모든 후보 크기를 병합하고 중복을 제거합니다.
    candidates = build_patch_candidates(periods, (img_h, img_w), textile_sizes)

    # Step 4) 각 후보 크기마다 오프셋을 탐색하고 패치 점수를 계산합니다.
    candidate_results = []
    for candidate in candidates:
        patch, crop, parts = select_patch_at_size(
            img,
            candidate["size"],
            patch_scorer,
            offset_step=offset_step,
            search_hw=candidate["search_hw"],
        )
        candidate_results.append(
            {
                "patch": patch,
                "crop": crop,
                "score_parts": parts,
                "candidate": candidate,
            }
        )

    # Step 5) 결합 손실이 최소인 전역 최적 패치를 선택합니다.
    candidate_results.sort(key=lambda item: item["score_parts"]["total"])
    best = candidate_results[0]
    info = {
        "vgg_period": (py_vgg, px_vgg),
        "fft_period": (py_fft, px_fft) if use_fft else None,
        "textile_size_candidates": textile_sizes,
        "candidate": best["candidate"],
        "candidate_results": candidate_results,
        "num_candidates": len(candidate_results),
    }
    return best["patch"], best["crop"], info


# =====================================================================
# 출력 생성 유틸
# =====================================================================
def repeat_tile(tile, out_h, out_w):
    """선택 패치를 주기적으로 반복해 최종 출력 크기로 만듭니다."""
    _, _, tile_h, tile_w = tile.shape
    repeats_h = math.ceil(out_h / tile_h)
    repeats_w = math.ceil(out_w / tile_w)
    tiled = tile.repeat(1, 1, repeats_h, repeats_w)
    return tiled[:, :, :out_h, :out_w].contiguous()


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def save_candidate_patches(candidate_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for idx, item in enumerate(candidate_results, start=1):
        candidate = item["candidate"]
        oy, ox, patch_h, patch_w = item["crop"]
        source = _safe_name(candidate["source"])
        path = os.path.join(
            output_dir,
            f"{idx:02d}_{source}_{patch_h}x{patch_w}_y{oy}_x{ox}.png",
        )
        save_image(item["patch"], path)
        saved.append(path)
    return saved


def format_score_parts(parts):
    base = (
        f"style={parts['style_loss']:.4f} "
        f"RTV={parts['rtv_loss']:.4f} "
        f"area={parts['area']} "
        f"areaPenalty={parts['area_loss']:.4f} "
    )
    if parts["textile_score"] is None:
        return base + f"score={parts['total']:.4f}"
    return (
        base
        + f"TexTile={parts['textile_score']:.4f} "
        + f"TexTileLoss={parts['textile_loss']:.4f} "
        + f"score={parts['total']:.4f}"
    )


# =====================================================================
# CLI 실행 진입점
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="seamless_out.png")
    parser.add_argument("--patch-output", default="min_patch.png")
    parser.add_argument("--candidate-output-dir", default="candidate_patches")
    parser.add_argument(
        "--out-size",
        type=parse_size,
        default=(1024, 1024),
        help="Final output size as HxW, e.g. 1024x1024.",
    )
    parser.add_argument(
        "--no-fft",
        action="store_true",
        help="Disable FFT power-spectrum period candidates and use VGG only.",
    )
    parser.add_argument(
        "--textile-ckpt",
        default=default_textile_checkpoint(),
        help="Path to the pretrained TexTile .pth or TorchScript model.",
    )
    parser.add_argument(
        "--no-textile",
        action="store_true",
        help="Disable TexTile scoring, rotation search, and TexTile size search.",
    )
    parser.add_argument(
        "--textile-resolution",
        type=parse_size,
        default=(512, 512),
        help="TexTile input resize as HxW, e.g. 512x512.",
    )
    parser.add_argument(
        "--textile-tiles",
        type=int,
        default=2,
        help="Number of repeats per axis before TexTile scoring.",
    )
    parser.add_argument(
        "--textile-lambda",
        type=float,
        default=TEXTILE_LOGIT_SCALE,
        help="Lambda in sigmoid(lambda * M(tile(I, (2, 2)))).",
    )
    parser.add_argument(
        "--textile-size-grid",
        type=int,
        default=16,
        help="Number of sampled crop sizes per axis for TexTile size search.",
    )
    parser.add_argument(
        "--textile-size-top-k",
        type=int,
        default=4,
        help="Number of TexTile crop-size candidates kept for patch search.",
    )
    parser.add_argument(
        "--rotation-step",
        type=float,
        default=5.0,
        help="Coarse angle step for argmax TexTile(Rotate(I, theta)).",
    )
    parser.add_argument(
        "--rotation-refine-step",
        type=float,
        default=1.0,
        help="Refinement angle step around the best coarse rotation.",
    )
    parser.add_argument(
        "--no-rotation-search",
        action="store_true",
        help="Disable TexTile rotation alignment.",
    )
    parser.add_argument(
        "--offset-step",
        type=int,
        default=8,
        help="Grid step for crop-position search.",
    )
    parser.add_argument(
        "--rtv-weight",
        type=float,
        default=1.0,
        help="Weight for RTV loss during patch selection.",
    )
    parser.add_argument(
        "--textile-weight",
        type=float,
        default=1.0,
        help="Weight for TexTile loss during patch selection.",
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=1.0,
        help="Weight for VGG Gram style loss during patch selection.",
    )
    parser.add_argument(
        "--area-weight",
        type=float,
        default=1e-4,
        help="Tiny weight for log patch area penalty during patch selection.",
    )
    parser.add_argument(
        "--style-resolution",
        type=parse_size,
        default=(256, 256),
        help="Resize used before VGG style feature extraction.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"fixed VGG layer = {VGG_LAYER}")

    if not _HAS_TV:
        raise RuntimeError(
            "torchvision is required. This script uses VGG19 layer-20 features."
        )

    # Step 0) 입력 이미지를 텐서(NCHW, [0,1])로 로드합니다.
    img = load_image(args.input, device, max_side=None)
    print(f"input = {tuple(img.shape)}")

    # Step 0.1) TexTile 스코어러를 구성합니다. (옵션)
    textile_scorer = None
    if not args.no_textile:
        textile_scorer = TextileTileabilityScorer(
            model_path=args.textile_ckpt,
            resolution=args.textile_resolution,
            number_tiles=args.textile_tiles,
            logit_scale=args.textile_lambda,
            device=device,
        )
        print(
            f"TexTile scorer = {args.textile_ckpt} "
            f"(lambda={args.textile_lambda})"
        )

    rotation_info = {
        "angle": 0.0,
        "score": None,
        "crop": (0, 0, img.shape[-2], img.shape[-1]),
        "evaluations": [],
    }

    # Step 0.2) TexTile 회전 탐색으로 입력을 정렬합니다. (옵션)
    if textile_scorer is not None and not args.no_rotation_search:
        img, rotation_info = find_best_rotation(
            img,
            textile_scorer,
            coarse_step=args.rotation_step,
            refine_step=args.rotation_refine_step,
            crop_valid=True,
        )
        print(
            "[0] rotation argmax TexTile(Rotate(I, theta)) = "
            f"{rotation_info['angle']:.2f} deg "
            f"TexTile={rotation_info['score']:.4f} "
            f"aligned_shape={tuple(img.shape)}"
        )
    elif args.no_rotation_search:
        print("[0] rotation search disabled")
    else:
        print("[0] rotation search skipped because TexTile is disabled")

    # Step 1) 패치 선택에 사용할 결합 점수 함수를 준비합니다.
    style_fn = VGGStyleLayer20(device, resolution=args.style_resolution)
    style_target_gram = style_fn.image_gram(img).detach()
    patch_scorer = TileabilityLoss(
        p=4,
        textile_scorer=textile_scorer,
        style_fn=style_fn,
        style_target_gram=style_target_gram,
        rtv_weight=args.rtv_weight,
        textile_weight=args.textile_weight,
        style_weight=args.style_weight,
        area_weight=args.area_weight,
    )

    # Step 2) 후보 크기/위치를 평가해 최적 패치를 추출합니다.
    act_fn = VGGActivations(device)
    patch, (oy, ox, patch_h, patch_w), extract_info = extract_tile_patch(
        img,
        act_fn,
        patch_scorer,
        textile_scorer=textile_scorer,
        offset_step=args.offset_step,
        use_fft=not args.no_fft,
        textile_size_grid=args.textile_size_grid,
        textile_size_top_k=args.textile_size_top_k,
    )

    print(f"[1] VGG20 period (py, px) = {extract_info['vgg_period']}")
    if extract_info["fft_period"] is not None:
        print(f"[1] FFT period (py, px) = {extract_info['fft_period']}")
    if extract_info["textile_size_candidates"]:
        best_size = extract_info["textile_size_candidates"][0]
        print(
            "[1] TexTile crop-size argmax = "
            f"{best_size['size']} "
            f"TexTile={best_size['textile_size_score']:.4f}"
        )

    selected = extract_info["candidate"]
    selected_parts = extract_info["candidate_results"][0]["score_parts"]
    print(
        f"[2] selected candidate = {selected['source']} "
        f"period={selected['period']} size={selected['size']} "
        f"from {extract_info['num_candidates']} candidates"
    )
    print(
        f"[2] patch @ (y={oy}, x={ox}) size=({patch_h}, {patch_w}) "
        f"{format_score_parts(selected_parts)}"
    )

    # Step 2.1) 선택 패치와 순위별 후보 패치를 모두 저장합니다.
    save_image(patch, args.patch_output)
    candidate_paths = save_candidate_patches(
        extract_info["candidate_results"],
        args.candidate_output_dir,
    )
    print(
        f"[2] saved {len(candidate_paths)} candidate patches "
        f"to {args.candidate_output_dir}"
    )

    # Step 3) 선택 패치를 단순 반복해 최종 출력 이미지를 생성합니다.
    out_h, out_w = args.out_size
    result = repeat_tile(patch, out_h, out_w)
    print(f"[3] repeated tile result = {tuple(result.shape)}")

    save_image(result, args.output)
    print(
        "saved: "
        f"{args.patch_output}, {args.candidate_output_dir}, {args.output}"
    )


if __name__ == "__main__":
    main()
