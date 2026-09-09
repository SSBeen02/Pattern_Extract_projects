import math
import os
from typing import Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


AXIS_PROFILE_MAX_K = 100


def normalize_to_u8(values: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.7) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [percentile_low, percentile_high])
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def normalize_01(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def safe_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def safe_bold_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        return safe_font(size)


def draw_text_segments(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    segments: List[Tuple[str, ImageFont.ImageFont]],
    fill: Tuple[int, int, int],
) -> None:
    x, y = xy
    for text, font in segments:
        draw.text((x, y), text, fill=fill, font=font)
        box = draw.textbbox((x, y), text, font=font)
        x = box[2]


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
    canvas_w, canvas_h = 1200, 840
    plot_w, plot_h = 980, 260
    left = 115
    top1, top2 = 105, 490
    bg = (255, 255, 255)
    axis_color = (35, 35, 35)
    grid_color = (220, 220, 220)
    amp_color = (28, 126, 210)
    peak_color = (0, 0, 0)
    text_color = (20, 20, 20)

    image = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(image)
    title_font = safe_font(24)
    font = safe_font(18)
    small_font = safe_font(15)

    title = f"Axis-wise 1D FFT Period Functions: {input_name}"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((canvas_w - (title_box[2] - title_box[0])) / 2, 28), title, fill=text_color, font=title_font)

    def plot_one(top: int, amps: np.ndarray, peaks: List[Any], title_text: str, xlabel: str) -> None:
        x0, y0 = left, top
        x1, y1 = left + plot_w, top + plot_h
        max_k = float(ks[-1])
        draw.rectangle([x0, y0, x1, y1], outline=axis_color, width=1)
        for i in range(1, 5):
            yy = y1 - i * plot_h / 5.0
            draw.line([(x0, yy), (x1, yy)], fill=grid_color, width=1)
            draw.text((x0 - 45, yy - 8), f"{i / 5:.1f}", fill=text_color, font=small_font)
        for k in [10, 30, 50, 70, 90, 100]:
            if k > max_k:
                continue
            xx = x0 + (k - 1) / (max_k - 1) * plot_w
            draw.line([(xx, y0), (xx, y1)], fill=grid_color, width=1)
            draw.text((xx - 14, y1 + 8), str(k), fill=text_color, font=small_font)
        draw.text((x0 - 45, y1 - 8), "0", fill=text_color, font=small_font)
        draw.text((x0 - 55, y0 - 8), "1.0", fill=text_color, font=small_font)
        draw.text((x0, y0 - 35), title_text, fill=text_color, font=font)
        draw.text((x0 + plot_w / 2 - 82, y1 + 38), xlabel, fill=text_color, font=small_font)
        draw.text((18, y0 + plot_h / 2 - 25), "Normalized\nvalue", fill=text_color, font=small_font)

        normalized = normalize_01(amps)
        pts: List[Tuple[float, float]] = []
        for idx, value in enumerate(normalized):
            k = float(ks[idx])
            xx = x0 + (k - 1.0) / (max_k - 1.0) * plot_w
            yy = y1 - float(value) * plot_h
            pts.append((xx, yy))
        draw.line(pts, fill=amp_color, width=2)

        for peak_idx, peak in enumerate(peaks):
            peak_x_pos = x0 + (peak.k - 1.0) / (max_k - 1.0) * plot_w
            nearest_idx = int(np.argmin(np.abs(ks - peak.k)))
            peak_y_pos = y1 - float(normalized[nearest_idx]) * plot_h
            radius = 3
            draw.ellipse(
                [peak_x_pos - radius, peak_y_pos - radius, peak_x_pos + radius, peak_y_pos + radius],
                fill=peak_color,
            )
            label = f"#{peak_idx + 1}: k={peak.k:.1f}"
            label_box = draw.textbbox((0, 0), label, font=small_font)
            label_w = label_box[2] - label_box[0]
            label_x = peak_x_pos + 10
            if label_x + label_w > x1 - 10:
                label_x = peak_x_pos - label_w - 10
            label_y = max(y0 + 8, peak_y_pos - 28 + peak_idx * 30)
            if peak_idx == 1:
                label_y += 18
            if label_y < y0 + 54 and label_x > x1 - 280:
                label_y = y0 + 64 + peak_idx * 30
            draw.text((label_x, label_y), label, fill=peak_color, font=small_font)
        draw.line([(x1 - 235, y0 + 18), (x1 - 185, y0 + 18)], fill=amp_color, width=2)
        draw.text((x1 - 175, y0 + 9), "amplitude", fill=text_color, font=small_font)

    plot_one(top1, amp_x, peaks_x, "X-axis", "kx")
    plot_one(top2, amp_y, peaks_y, "Y-axis", "ky")
    image.save(out_path)


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


def ncc_map_to_heatmap(values: np.ndarray) -> Image.Image:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Image.new("RGB", (max(1, values.shape[1]), max(1, values.shape[0])), (0, 0, 0))
    lo, hi = np.percentile(finite, [1.0, 99.7])
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    safe_values = np.nan_to_num(values, nan=lo, posinf=hi, neginf=lo)
    scaled = np.clip((safe_values - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    r = np.clip((scaled - 0.5) / 0.5, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(scaled - 0.5) / 0.5, 0.0, 1.0)
    b = np.clip((0.5 - scaled) / 0.5, 0.0, 1.0)
    rgb = np.dstack([r, g, b]) * 255.0
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def draw_grid_patch_preview(
    image: Image.Image,
    base_dx: float,
    base_dy: float,
    patch_box: Tuple[int, int, int, int],
) -> Image.Image:
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis, "RGBA")
    w, h = vis.size
    cx = w / 2.0
    cy = h / 2.0
    grid_color = (138, 78, 255, 185)
    center_color = (255, 0, 0, 190)
    box_color = (255, 190, 0, 230)

    x = cx
    while x >= 0:
        draw.line([(x, 0), (x, h)], fill=grid_color, width=3)
        x -= base_dx
    x = cx + base_dx
    while x < w:
        draw.line([(x, 0), (x, h)], fill=grid_color, width=3)
        x += base_dx

    y = cy
    while y >= 0:
        draw.line([(0, y), (w, y)], fill=grid_color, width=3)
        y -= base_dy
    y = cy + base_dy
    while y < h:
        draw.line([(0, y), (w, y)], fill=grid_color, width=3)
        y += base_dy

    draw.line([(cx, 0), (cx, h)], fill=center_color, width=1)
    draw.line([(0, cy), (w, cy)], fill=center_color, width=1)
    for inset in range(3):
        draw.rectangle(
            [patch_box[0] - inset, patch_box[1] - inset, patch_box[2] + inset, patch_box[3] + inset],
            outline=box_color,
        )
    return vis


def save_patch_box_preview(
    image: Image.Image,
    patch_sizes: List[Tuple[int, int, int, int]],
    out_path: str,
) -> None:
    box_preview = image.copy()
    box_draw = ImageDraw.Draw(box_preview)
    box_w, box_h = box_preview.size
    box_cx = box_w / 2.0
    box_cy = box_h / 2.0
    box_draw.line([(box_cx, 0), (box_cx, box_h)], fill=(255, 0, 0), width=2)
    box_draw.line([(0, box_cy), (box_w, box_cy)], fill=(255, 0, 0), width=2)
    box_colors = [
        (98, 78, 255),
        (0, 210, 80),
        (255, 170, 0),
        (0, 180, 255),
        (190, 90, 255),
        (255, 80, 190),
        (80, 120, 255),
        (0, 170, 130),
    ]
    label_font = safe_font(max(12, int(round(min(box_w, box_h) / 70))))
    for idx, (dx_mult, dy_mult, patch_w, patch_h) in enumerate(patch_sizes):
        left = int(round(box_cx - patch_w / 2.0))
        top = int(round(box_cy - patch_h / 2.0))
        right = left + patch_w
        bottom = top + patch_h
        color = box_colors[idx % len(box_colors)]
        for inset in range(2):
            box_draw.rectangle([left - inset, top - inset, right + inset, bottom + inset], outline=color)
        box_draw.text((left + 4, top + 4), f"{dx_mult}x,{dy_mult}y", fill=color, font=label_font)
    box_preview.save(out_path)


def save_patch_montage(
    base_dx: float,
    base_dy: float,
    montage_items: List[Tuple[int, int, Image.Image, Image.Image, Image.Image, Image.Image, float]],
    out_path: str,
) -> None:
    cell_w, cell_h = 1800, 860
    label_h = 70
    margin = 22
    title_h = 54
    canvas_w = margin * 2 + cell_w * 4
    canvas_h = margin * 2 + title_h + (cell_h + label_h) * 4
    montage = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(montage)
    title_font = safe_font(20)
    label_font = safe_font(28)
    label_bold_font = safe_bold_font(34)
    title = f"Period patch comparison: dx={base_dx:.1f}px, dy={base_dy:.1f}px"
    draw.text((margin, margin), title, fill=(0, 0, 0), font=title_font)
    panel_titles = ["grid", "patch", "Extended patch", "Original crop"]
    panel_font = safe_font(54)
    panel_w = (cell_w - 54) // 4
    panel_h = cell_h - 56
    ranked_indices = sorted((i for i, item in enumerate(montage_items) if np.isfinite(item[-1])),
                            key=lambda i: montage_items[i][-1], reverse=True)
    ranks = {i: rank for rank, i in enumerate(ranked_indices, start=1)}
    for idx, (dx_mult, dy_mult, grid_preview, patch, extended, reference, score) in enumerate(montage_items):
        col = idx % 4
        row = idx // 4
        cell_left = margin + col * cell_w
        cell_top = margin + title_h + row * (cell_h + label_h)
        draw.rectangle([cell_left, cell_top, cell_left + cell_w, cell_top + cell_h], outline=(0, 0, 0), width=3)
        panels = [grid_preview, patch, extended, reference]
        for panel_idx, panel in enumerate(panels):
            panel_left = cell_left + 10 + panel_idx * (panel_w + 10)
            panel_top = cell_top + 44
            preview = panel.copy()
            preview.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
            paste_x = panel_left + (panel_w - preview.size[0]) // 2
            paste_y = panel_top + (panel_h - preview.size[1]) // 2
            montage.paste(preview, (paste_x, paste_y))
            draw.rectangle([panel_left, panel_top, panel_left + panel_w, panel_top + panel_h], outline=(150, 150, 150), width=1)
            draw.text((panel_left + 8, cell_top + 6), panel_titles[panel_idx], fill=(80, 80, 80), font=panel_font)
        draw_text_segments(
            draw,
            (cell_left + 16, cell_top + cell_h + 12),
            [
                ("dx x", label_font),
                (str(dx_mult), label_bold_font),
                (", dy x", label_font),
                (str(dy_mult), label_bold_font),
                (f" ({patch.size[0]}x{patch.size[1]})", label_font),
                ((f"  NCC={score:.6f}  Rank={ranks[idx]}" if np.isfinite(score) else "  NCC=N/A"), label_font),
            ],
            fill=(0, 0, 0),
        )
    montage.save(out_path)
