from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "outputs" / "patch_ncc_summary"
OUTPUT_PDF = ROOT / "output" / "pdf" / "patch_ncc_summary.pdf"
TITLE = "patch_ncc_summary"


def draw_centered_text(c, text, y, page_width, font="Helvetica-Bold", size=34):
    c.setFont(font, size)
    text_width = stringWidth(text, font, size)
    c.drawString((page_width - text_width) / 2, y, text)


def build_pdf():
    images = sorted(INPUT_DIR.glob("*_period_patches_16.png"))
    if not images:
        raise FileNotFoundError(f"No montage images found in {INPUT_DIR}")

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    max_page_width = 5200
    margin = 36
    header_h = 120
    footer_h = 42

    c = canvas.Canvas(str(OUTPUT_PDF))
    c.setTitle(TITLE)
    c.setAuthor("Codex")
    c.setSubject("Patch NCC summary montages")

    for page_num, image_path in enumerate(images, 1):
        with Image.open(image_path) as im:
            img_w, img_h = im.size

        page_w = min(max_page_width, img_w + 2 * margin)
        draw_w = page_w - 2 * margin
        scale = draw_w / img_w
        draw_h = img_h * scale
        page_h = draw_h + header_h + footer_h

        c.setPageSize((page_w, page_h))
        c.setFillColor(colors.white)
        c.rect(0, 0, page_w, page_h, fill=1, stroke=0)

        draw_centered_text(c, TITLE, page_h - 52, page_w)
        case_name = image_path.name.replace("_period_patches_16.png", "")
        draw_centered_text(c, case_name, page_h - 88, page_w, size=22)

        c.drawImage(
            ImageReader(str(image_path)),
            margin,
            footer_h,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            mask="auto",
        )

        c.setFont("Helvetica", 16)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawRightString(page_w - margin, 18, f"{page_num} / {len(images)}")
        c.showPage()

    c.save()
    return OUTPUT_PDF


if __name__ == "__main__":
    print(build_pdf())
