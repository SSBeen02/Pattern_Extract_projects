import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List

from PIL import Image
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


OUTPUT_TYPES = [
    ("axis_period_functions", "Axis Period"),
    ("frequency_domain_peaks", "Frequency Domain"),
    ("spatial_domain_peaks", "Spatial Domain"),
]


@dataclass
class CaseImages:
    case_id: str
    paths: Dict[str, str]


def collect_cases(output_dir: str) -> List[CaseImages]:
    pattern = re.compile(
        r"^(?P<case>.+)_(?P<kind>axis_period_functions|frequency_domain_peaks|spatial_domain_peaks)\.png$"
    )
    grouped: Dict[str, Dict[str, str]] = {}
    for name in sorted(os.listdir(output_dir)):
        match = pattern.match(name)
        if not match:
            continue
        grouped.setdefault(match.group("case"), {})[match.group("kind")] = os.path.join(output_dir, name)

    required = {kind for kind, _label in OUTPUT_TYPES}
    return [
        CaseImages(case_id=case_id, paths=paths)
        for case_id, paths in sorted(grouped.items())
        if required.issubset(paths)
    ]


def draw_centered_text(c: canvas.Canvas, text: str, x: float, y: float, font_name: str, font_size: float) -> None:
    c.setFont(font_name, font_size)
    c.drawString(x - stringWidth(text, font_name, font_size) / 2.0, y, text)


def draw_image_fit(
    c: canvas.Canvas,
    path: str,
    x: float,
    y: float,
    w: float,
    h: float,
    valign: str = "center",
) -> None:
    with Image.open(path) as img:
        img_w, img_h = img.size
    scale = min(w / img_w, h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    draw_x = x + (w - draw_w) / 2.0
    if valign == "bottom":
        draw_y = y
    elif valign == "top":
        draw_y = y + h - draw_h
    else:
        draw_y = y + (h - draw_h) / 2.0
    c.drawImage(path, draw_x, draw_y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")


def build_pdf(output_dir: str, pdf_path: str, title: str = "Period FFT Outputs Summary", first_page_note: str = "") -> int:
    cases = collect_cases(output_dir)
    if not cases:
        raise RuntimeError(f"No complete image triplets found in {output_dir}")

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    page_w, page_h = 1700, 1600
    margin_x = 32
    top_margin = 28
    bottom_margin = 40
    gap = 40
    title_y = page_h - top_margin
    axis_caption_y = page_h - 96
    axis_y = 830
    axis_h = 650
    bottom_caption_y = 782
    bottom_y = bottom_margin
    bottom_h = 700
    bottom_w = (page_w - margin_x * 2 - gap) / 2.0

    c = canvas.Canvas(pdf_path, pagesize=(page_w, page_h))
    c.setTitle(title)

    for page_num, case in enumerate(cases, start=1):
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin_x, title_y, case.case_id)
        if page_num == 1 and first_page_note:
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(page_w / 2.0, title_y, first_page_note)
        c.setFont("Helvetica", 11)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawRightString(page_w - margin_x, title_y, f"{page_num} / {len(cases)}")

        c.setFillColor(colors.black)
        draw_centered_text(c, "Axis Period", page_w / 2.0, axis_caption_y, "Helvetica-Bold", 12)
        draw_image_fit(
            c,
            case.paths["axis_period_functions"],
            margin_x,
            axis_y,
            page_w - margin_x * 2,
            axis_h,
        )

        freq_x = margin_x
        spatial_x = margin_x + bottom_w + gap
        draw_centered_text(c, "Frequency Domain", freq_x + bottom_w / 2.0, bottom_caption_y, "Helvetica-Bold", 12)
        draw_centered_text(c, "Spatial Domain", spatial_x + bottom_w / 2.0, bottom_caption_y, "Helvetica-Bold", 12)
        draw_image_fit(c, case.paths["frequency_domain_peaks"], freq_x, bottom_y, bottom_w, bottom_h, "bottom")
        draw_image_fit(c, case.paths["spatial_domain_peaks"], spatial_x, bottom_y, bottom_w, bottom_h, "bottom")
        c.showPage()

    c.save()
    return len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a PDF summary from period FFT output images.")
    parser.add_argument("--output-dir", default=None, help="Folder containing generated period FFT output images.")
    parser.add_argument("--pdf", default=None, help="PDF path to write.")
    parser.add_argument("--title", default="Period FFT Outputs Summary", help="PDF document title.")
    parser.add_argument("--first-page-note", default="", help="Text to show at the top of the first page.")
    args = parser.parse_args()

    root = os.path.abspath(os.path.dirname(__file__))
    out_dir = args.output_dir or os.path.join(root, "period_fft_outputs")
    pdf_out = args.pdf or os.path.join(root, "output", "pdf", "period_fft_outputs_summary.pdf")
    count = build_pdf(out_dir, pdf_out, title=args.title, first_page_note=args.first_page_note)
    print(f"pdf={pdf_out}")
    print(f"cases={count}")
