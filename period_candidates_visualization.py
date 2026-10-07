import math
import argparse
import csv
import os
from typing import Any, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_to_u8(values: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.7) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [percentile_low, percentile_high])
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def safe_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def format_truncated(value: float, decimal_places: int = 1) -> str:
    """소수 자릿수를 반올림하지 않고 버린 뒤, 불필요한 끝자리 0을 제거한다."""
    factor = 10 ** decimal_places
    truncated = math.trunc(float(value) * factor) / factor
    formatted = f"{truncated:.{decimal_places}f}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def visual_scale(width: int, height: int) -> float:
    return max(1.0, min(width, height) / 768.0)


def period_points_in_image(
    start_x: float,
    start_y: float,
    dx: float,
    dy: float,
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    period = math.hypot(dx, dy)
    if period < 1e-6:
        return points

    max_steps = int(math.ceil(math.hypot(width, height) / period)) + 2
    for step in range(-max_steps, max_steps + 1):
        x = start_x + step * dx
        y = start_y + step * dy
        if 0 <= x < width and 0 <= y < height:
            point = (int(round(x)), int(round(y)))
            if point not in points:
                points.append(point)
    return points


def draw_circle_outline(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    radius: int,
    color: Tuple[int, ...],
    width: int,
) -> None:
    for i in range(width):
        draw.ellipse([x - radius - i, y - radius - i, x + radius + i, y + radius + i], outline=color)


def draw_filled_circle(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    radius: int,
    color: Tuple[int, ...],
) -> None:
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    color: Tuple[int, ...],
    line_width: int,
    head_len: int,
) -> None:
    draw.line([p0, p1], fill=color, width=line_width)
    angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    spread = math.radians(28)
    left = (
        int(round(p1[0] - head_len * math.cos(angle - spread))),
        int(round(p1[1] - head_len * math.sin(angle - spread))),
    )
    right = (
        int(round(p1[0] - head_len * math.cos(angle + spread))),
        int(round(p1[1] - head_len * math.sin(angle + spread))),
    )
    draw.polygon([p1, left, right], fill=color)


def save_axis_period_functions(
    ks: np.ndarray,
    amp_x: np.ndarray,
    amp_y: np.ndarray,
    peaks_x: List[Any],
    peaks_y: List[Any],
    out_path: str,
    input_name: str,
) -> None:
    # ks: FFT 주파수 인덱스 배열, amp_x/amp_y: 각 축의 원시 amplitude 배열이다.
    # peaks_x/peaks_y에 담긴 검출 피크 위치를 그래프에 표시하고 결과 이미지를 저장한다.
    # amplitude에는 로그 변환이나 정규화를 적용하지 않으며, 표시 좌표만 축 범위에 맞춰 환산한다.

    # 두 개의 축 그래프와 제목을 담을 전체 이미지 및 각 그래프의 크기를 정한다.
    canvas_w, canvas_h = 1200, 840
    plot_w, plot_h = 980, 260
    left = 115
    top1, top2 = 105, 490

    # 그래프에서 사용하는 색상들. 배경, 축, 격자, amplitude 선, 피크 표식을 분리해 가독성을 높인다.
    bg = (255, 255, 255)
    axis_color = (35, 35, 35)
    grid_color = (220, 220, 220)
    amp_color = (28, 126, 210)
    peak_color = (0, 0, 0)
    text_color = (20, 20, 20)

    # PIL 이미지와 그리기 객체를 준비하고, 제목/축 라벨에 사용할 글꼴을 불러온다.
    image = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(image)
    title_font = safe_font(24)
    font = safe_font(18)
    small_font = safe_font(15)

    # 전체 제목은 입력 파일명을 포함하며, 텍스트 너비를 계산해 캔버스 가운데 배치한다.
    title = f"Axis-wise 1D FFT Period Functions: {input_name}"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((canvas_w - (title_box[2] - title_box[0])) / 2, 28), title, fill=text_color, font=title_font)

    # X 또는 Y 한 축의 데이터를 실제 그래프 영역에 그린다.
    # top은 그래프의 세로 위치이고, 나머지 인자는 해당 축의 amplitude/피크/라벨이다.
    def plot_one(top: int, amps: np.ndarray, peaks: List[Any], title_text: str, xlabel: str) -> None:
        # 왼쪽 위(x0, y0)와 오른쪽 아래(x1, y1)로 그래프 사각형을 정의한다.
        x0, y0 = left, top
        x1, y1 = left + plot_w, top + plot_h

        # x축은 FFT 인덱스 k의 범위를 사용한다. 마지막 ks 값이 그래프의 오른쪽 끝이 된다.
        max_k = float(ks[-1])

        # NaN/inf가 최댓값 계산을 망치지 않도록 유한한 amplitude만 골라 y축 상한을 구한다.
        # 눈금 간격을 10의 배수로 올림해 데이터 최댓값보다 큰 깔끔한 축 범위를 만든다.
        finite_amps = amps[np.isfinite(amps)]
        data_max = float(np.max(finite_amps)) if finite_amps.size else 0.0
        tick_step = max(10, math.ceil((data_max * 1.05 / 5.0) / 10.0) * 10)
        y_max = tick_step * 5

        # 그래프 테두리와 y축 0~y_max 구간의 수평 격자/실제 amplitude 눈금을 그린다.
        draw.rectangle([x0, y0, x1, y1], outline=axis_color, width=1)
        for i in range(1, 6):
            yy = y1 - i * plot_h / 5.0
            draw.line([(x0, yy), (x1, yy)], fill=grid_color, width=1)
            draw.text((x0 - 70, yy - 8), f"{tick_step * i:,}", fill=text_color, font=small_font)

        # 관심 있는 FFT 인덱스에 세로 격자선을 그리고 k 값을 표시한다.
        for k in [10, 30, 50, 70, 90, 100]:
            if k > max_k:
                continue
            xx = x0 + (k - 1) / (max_k - 1) * plot_w
            draw.line([(xx, y0), (xx, y1)], fill=grid_color, width=1)
            draw.text((xx - 14, y1 + 8), str(k), fill=text_color, font=small_font)
        # 세로축의 0 위치, 그래프 제목, 가로축 이름을 표시한다.
        draw.text((x0 - 70, y1 - 8), "0", fill=text_color, font=small_font)
        draw.text((x0, y0 - 35), title_text, fill=text_color, font=font)
        draw.text((x0 + plot_w / 2 - 82, y1 + 38), xlabel, fill=text_color, font=small_font)
        

        # 각 (k, amplitude) 표본을 픽셀 좌표로 변환한다.
        # x는 k 범위를 그래프 너비에 선형 대응시키고, y는 원시 amplitude를 [0, y_max]에 대응시킨다.
        # 데이터 자체에는 정규화를 적용하지 않으며 y 좌표를 계산할 때만 표시 범위로 나눈다.
        pts: List[Tuple[float, float]] = []
        for idx, value in enumerate(amps):
            k = float(ks[idx])
            xx = x0 + (k - 1.0) / (max_k - 1.0) * plot_w
            yy = y1 - float(value) / y_max * plot_h
            pts.append((xx, yy))
        # 변환된 표본들을 선으로 연결해 축별 amplitude 곡선을 그린다.
        draw.line(pts, fill=amp_color, width=2)

        # 선택된 피크의 k 위치와 가장 가까운 정수 FFT 표본의 amplitude 위치에 검은 점을 표시한다.
        for peak_idx, peak in enumerate(peaks):
            peak_x_pos = x0 + (peak.k - 1.0) / (max_k - 1.0) * plot_w
            nearest_idx = int(np.argmin(np.abs(ks - peak.k)))
            peak_y_pos = y1 - float(amps[nearest_idx]) / y_max * plot_h
            radius = 3
            draw.ellipse(
                [peak_x_pos - radius, peak_y_pos - radius, peak_x_pos + radius, peak_y_pos + radius],
                fill=peak_color,
            )
            # 피크 번호와 소수점 한 자리의 k를 라벨로 만들고, 그래프 오른쪽을 넘지 않게 좌우를 선택한다.
            label = f"#{peak_idx + 1}: k={peak.k:.1f}"
            label_box = draw.textbbox((0, 0), label, font=small_font)
            label_w = label_box[2] - label_box[0]
            label_x = peak_x_pos + 10
            if label_x + label_w > x1 - 10:
                label_x = peak_x_pos - label_w - 10
            # 곡선에 라벨이 겹치는 것을 줄이기 위해 피크 높이를 기준으로 라벨을 배치한다.
            # 상단 경계를 넘지 않도록 제한하고, 상단 우측에서 겹칠 가능성이 있으면 아래로 옮긴다.
            label_y = max(y0 + 8, peak_y_pos - 28 + peak_idx * 30)
            if peak_idx == 1:
                label_y += 18
            if label_y < y0 + 54 and label_x > x1 - 280:
                label_y = y0 + 64 + peak_idx * 30
            draw.text((label_x, label_y), label, fill=peak_color, font=small_font)
        # 곡선 색과 'amplitude' 텍스트를 범례처럼 표시한다.
        draw.line([(x1 - 235, y0 + 18), (x1 - 185, y0 + 18)], fill=amp_color, width=2)
        draw.text((x1 - 175, y0 + 9), "amplitude", fill=text_color, font=small_font)

    # X축 및 Y축 프로파일을 각각 위/아래 그래프에 그린 뒤 지정 경로에 PNG로 저장한다.
    plot_one(top1, amp_x, peaks_x, "X-axis", "kx")
    plot_one(top2, amp_y, peaks_y, "Y-axis", "ky")
    image.save(out_path)


''' 
def draw_frequency_domain_peaks(
    mag: np.ndarray,
    candidates: List[Any],
    out_path: str,
    visual_limit: int = 4,
) -> None:
    full_image = Image.fromarray(normalize_to_u8(mag), mode="L").convert("RGB")
    full_w, full_h = full_image.size
    full_cx, full_cy = full_w // 2, full_h // 2
    crop_size = min(512, full_w, full_h)
    crop_half = crop_size // 2
    crop_left = full_cx - crop_half
    crop_top = full_cy - crop_half
    crop = full_image.crop((crop_left, crop_top, crop_left + crop_size, crop_top + crop_size))
    shown = candidates[:visual_limit]
    font = safe_font(16)
    labels = [
        f"#{cand.candidate_id}: k=({cand.kx:.1f},{cand.ky:.1f}), "
        f"d=({cand.dx:.1f},{cand.dy:.1f}), period={cand.period_px:.1f}px"
        for cand in shown
    ]
    header_h = 10 + 22 * max(1, len(labels)) + 10
    image = Image.new("RGB", (crop_size, crop_size + header_h), (255, 255, 255))
    image.paste(crop, (0, header_h))
    draw = ImageDraw.Draw(image)
    w, h = image.size
    cx, cy = crop_size // 2, header_h + crop_size // 2
    line_width = 1
    radius = 5

    draw.line([(cx, header_h), (cx, header_h + crop_size)], fill=(255, 220, 0), width=max(1, line_width // 2))
    draw.line([(0, cy), (crop_size, cy)], fill=(255, 220, 0), width=max(1, line_width // 2))
    search_color = (0, 160, 255)
    search_min_k = 5
    search_max_k = min(AXIS_PROFILE_MAX_K, crop_size // 2)
    draw.line([(cx + search_min_k, cy), (cx + search_max_k, cy)], fill=search_color, width=2)
    draw.line([(cx, cy + search_min_k), (cx, cy + search_max_k)], fill=search_color, width=2)

    candidate_colors = [(98, 78, 255), (0, 210, 80), (255, 170, 0), (0, 210, 255)]
    for idx, cand in enumerate(shown):
        color = candidate_colors[min(idx, len(candidate_colors) - 1)]
        local_x = cand.peak_x - float(crop_left)
        local_y = cand.peak_y - float(crop_top) + header_h
        if 0 <= local_x < w and 0 <= local_y < h:
            draw_circle_outline(draw, int(round(local_x)), int(round(local_y)), radius, color, line_width)

    for idx, label in enumerate(labels):
        draw.text((12, 10 + idx * 22), label, fill=(0, 0, 0), font=font)

    image.save(out_path)
'''

def draw_spatial_domain_peaks(
    image: Image.Image,
    candidates: List[Any],
    out_path: str,
    visual_limit: int = 4,
) -> None:
    w, h = image.size
    scale = visual_scale(w, h)
    line_width = max(1, int(round(2.0 * scale)))
    point_radius = max(3, int(round(3.8 * scale)))
    head_len = max(9, int(round(12 * scale)))
    arrow_color = (255, 0, 0)
    shown = candidates[:visual_limit]
    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    for idx, cand in enumerate(shown):
        start_x = float(w // 2)
        start_y = float(h // 2)
        points = period_points_in_image(start_x, start_y, cand.dx, cand.dy, w, h)
        start = (int(round(start_x)), int(round(start_y)))
        end = (int(round(start_x + cand.dx)), int(round(start_y + cand.dy)))
        point_colors = [
            (98, 78, 255),
            (0, 210, 80),
            (255, 170, 0),
            (0, 210, 255),
            (190, 90, 255),
            (255, 80, 190),
            (80, 120, 255),
            (0, 170, 130),
        ]
        point_color = point_colors[min(idx, len(point_colors) - 1)]
        offset_step = point_radius * 6 + 4
        point_offsets = [
            (0, 0),
            (offset_step, offset_step),
            (-offset_step, offset_step),
            (offset_step, -offset_step),
            (-offset_step, -offset_step),
            (offset_step * 2, 0),
            (0, offset_step * 2),
            (-offset_step * 2, 0),
        ]
        point_offset = point_offsets[min(idx, len(point_offsets) - 1)]
        arrow_start = (start[0] + point_offset[0], start[1] + point_offset[1])
        arrow_end = (end[0] + point_offset[0], end[1] + point_offset[1])

        for x, y in points:
            draw_filled_circle(draw, x + point_offset[0], y + point_offset[1], point_radius, point_color)
        draw_arrow(draw, arrow_start, arrow_end, arrow_color, line_width, head_len)

    vis.save(out_path)


def _draw_score_montage(
    original: Image.Image,
    patches: List[Any],
    base_dx: float,
    base_dy: float,
    out_path: str,
    score_method: str,
) -> None:
    score_label = score_method.upper()
    # 캔버스 크기와 비율은 기존 4열×4행 레이아웃의 크기를 유지한다.
    cell_w, cell_h = 630, 260
    label_h = 58
    margin = 30
    title_h = 46
    canvas_original_slot_h = 260  # 기존 전체 캔버스 크기와 비율을 유지하기 위한 레이아웃 기준 높이
    original_preview_h = 520  # 실제 원본 미리보기는 기존 표시 크기의 2배
    section_title_h = 34
    grid_h = section_title_h + (cell_h + label_h) * 4
    canvas_w = margin * 2 + cell_w * 4
    canvas_h = margin * 3 + title_h + canvas_original_slot_h + grid_h
    montage = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(montage)
    title_font = safe_font(40)
    section_font = safe_font(32)
    label_font = safe_font(36)

    title = f"{score_label} patch comparison: dx={base_dx:.1f}px, dy={base_dy:.1f}px"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_y = margin
    draw.text((margin, title_y), title, fill=(0, 0, 0), font=title_font)

    # 한 개 몽타주의 NCC 설명과 Rank 정보를 좌측 상단에 제목 아래로 이어서 표시한다.
    result = patches[0] if patches else None
    if result is not None:
        row_data = result.metadata
        dx_mult, dy_mult = row_data["dx_mult"], row_data["dy_mult"]
        score, rank = row_data[f"{score_method}_score"], row_data["rank"]
        image_patch = result.image_patch
        
        comparison_title = ("VGG19 feature comparison"
                            if score_method == "vgg19" else "NCC similarity on the extended margin")
        comparison_title_box = draw.textbbox((0, 0), comparison_title, font=section_font)
        comparison_y = title_y + (title_box[3] - title_box[1]) + 30
        draw.text((margin, comparison_y), comparison_title, fill=(0, 0, 0), font=section_font)

        info = (f"Rank={rank if np.isfinite(score) else 'N/A'}  |  "
                f"dx x{dx_mult}, dy x{dy_mult}  |  "
                f"Patch={image_patch.width}x{image_patch.height}  |  "
                f"{score_label}={score:.6f}" if np.isfinite(score) else
                f"Rank=N/A  |  dx x{dx_mult}, dy x{dy_mult}  |  "
                f"Patch={image_patch.width}x{image_patch.height}  |  {score_label}=N/A")
        info_box = draw.textbbox((0, 0), info, font=label_font)
        info_y = comparison_y + (comparison_title_box[3] - comparison_title_box[1]) + 40
        draw.text((margin, info_y), info, fill=(0, 0, 0), font=label_font)
        header_bottom = info_y + (info_box[3] - info_box[1])
    else:
        header_bottom = title_y + (title_box[3] - title_box[1])

    # 원본 이미지는 상단 중앙에 두되, 왼쪽의 세 줄 텍스트 아래에서 시작하게 한다.
    original_y = max(margin + title_h, header_bottom + 16)
    if original is not None:
        original_preview = original.copy().convert("RGB")
        original_preview.thumbnail((canvas_w - margin * 2, original_preview_h), Image.Resampling.LANCZOS)
        original_x = (canvas_w - original_preview.width) // 2
        montage.paste(original_preview, (original_x, original_y))
        center_x = original_x + original_preview.width // 2
        center_y = original_y + original_preview.height // 2
        draw.line([(center_x, original_y), (center_x, original_y + original_preview.height)], fill=(255, 0, 0), width=2)
        draw.line([(original_x, center_y), (original_x + original_preview.width, center_y)], fill=(255, 0, 0), width=2)
        draw.rectangle(
            [original_x, original_y, original_x + original_preview.width, original_y + original_preview.height],
            outline=(210, 210, 210),
            width=1,
        )
        original_label = "Original image"
        label_box = draw.textbbox((0, 0), original_label, font=label_font)
        draw.text(((canvas_w - (label_box[2] - label_box[0])) // 2,
                   original_y + original_preview.height + 4), original_label, fill=(0, 0, 0), font=label_font)

    # 세 비교 이미지는 원본 이미지 아래에 가로 한 줄로 배치한다.
    if result is not None:
        image_patch, extended, reference = result.image_patch, result.extended, result.reference

        panel_gap = 24
        panel_margin = 60
        panel_w = (canvas_w - panel_margin * 2 - panel_gap * 2) // 3
        panel_lefts = [panel_margin + i * (panel_w + panel_gap) for i in range(3)]
        panel_top = original_y + (original_preview.height if original is not None else 0) + 72
        image_top = panel_top + 100
        image_h = canvas_h - image_top - 150
        preview_w = panel_w - 24

        panels = (
            (image_patch, "Patch"),
            (extended, "Extended patch"),
            (reference, "Original crop"),
        )
        for panel_left, (panel, panel_title) in zip(panel_lefts, panels):
            title_box = draw.textbbox((0, 0), panel_title, font=label_font)
            draw.text((panel_left + (panel_w - (title_box[2] - title_box[0])) // 2, panel_top + 50),
                      panel_title, fill=(60, 60, 60), font=label_font)
            box_top = image_top + 28
            box_bottom = box_top + image_h
            draw.rectangle([panel_left, box_top, panel_left + panel_w, box_bottom],
                           outline=(180, 180, 180), width=1)
            if panel is None:
                message = "out of size"
                bounds = draw.textbbox((0, 0), message, font=section_font)
                draw.text((panel_left + (panel_w - bounds[2] + bounds[0]) // 2,
                           box_top + (image_h - (bounds[3] - bounds[1])) // 2),
                          message, fill=(160, 60, 60), font=section_font)
            else:
                preview = panel.copy().convert("RGB")
                # thumbnail()은 원본보다 확대하지 않으므로, 패널을 채우도록 비율을 유지해 직접 리사이즈한다.
                scale = min(preview_w / preview.width, (image_h - 16) / preview.height)
                preview = preview.resize(
                    (max(1, round(preview.width * scale)), max(1, round(preview.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                paste_x = panel_left + (panel_w - preview.width) // 2
                paste_y = box_top + (image_h - preview.height) // 2
                montage.paste(preview, (paste_x, paste_y))

    montage.save(out_path)


def draw_ncc_montage(original, patches, base_dx, base_dy, out_path) -> None:
    """Render precomputed NCC scores."""
    _draw_score_montage(original, patches, base_dx, base_dy, out_path, "ncc")


def draw_vgg19_montage(original, patches, base_dx, base_dy, out_path) -> None:
    """Render precomputed VGG19 feature similarity scores."""
    _draw_score_montage(original, patches, base_dx, base_dy, out_path, "vgg19")


def save_period_patches(image, results, output_dir, score_method="ncc") -> None:
    """Save and visualize patches already generated and scored by fft2."""
    renderer = {"ncc": draw_ncc_montage, "vgg19": draw_vgg19_montage}[score_method]
    ensure_dir(output_dir)
    for result in results:
        row = result.metadata
        if row["score_method"] != score_method:
            raise ValueError("Patch score method does not match visualization method")
        dx_mult, dy_mult = row["dx_mult"], row["dy_mult"]
        if dx_mult == 1 and dy_mult == 1:
            result.image_patch.save(os.path.join(output_dir, "original_patch.png"))
        out_path = os.path.join(output_dir,
            f"{score_method}_montage_rank{row['rank']}_dx{dx_mult}_dy{dy_mult}.png")
        renderer(image, [result], row["dx"], row["dy"], out_path)
