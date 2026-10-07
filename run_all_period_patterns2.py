"""Run period extraction for every input image and collect outputs in one folder."""
import argparse
import contextlib
import io
from pathlib import Path
import re
import sys
import tempfile

from PIL import Image
import period_candidates_fft2 as fft


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_sort_key(path: Path) -> list[object]:
    """Sort paths by their numeric portions, so image2 precedes image10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.as_posix())]


def output_name(generated_name: str, index: int) -> str:
    """Map FFT output names to the requested two-digit-prefixed flat names."""
    if generated_name == "axis_period_functions.png":
        suffix = "axis.png"
    elif generated_name == "spatial_domain_peaks.png":
        suffix = "period.png"
    else:
        suffix = generated_name
    return f"{index:02d}_{suffix}"



def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Process all input patterns and save every result in one output folder."
    )
    parser.add_argument("--input-dir", "--input_dir", type=Path,
                        default=Path("input_pattern/aligned_1024ver"),
                        help="Input image folder (searched recursively). Relative paths use the script folder.")
    parser.add_argument("--output-dir", "--output_dir", type=Path,
                        default=Path("outputs/1006"),
                        help="Output folder. Relative paths use the script folder.")
    parser.add_argument("--score-method", "--score_method", choices=("ncc", "vgg19"),
                        default="ncc", help="Patch scoring method (default: ncc).")
    args = parser.parse_args()

    input_dir = (args.input_dir if args.input_dir.is_absolute() else root / args.input_dir).resolve()
    output_dir = (args.output_dir if args.output_dir.is_absolute() else root / args.output_dir).resolve()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist or is not a directory: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        parser.error("Output folder must be outside the input folder to avoid reprocessing generated images.")

    images = sorted(
        (path for path in input_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_sort_key,
    )
    if not images:
        raise FileNotFoundError(f"No input images found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"score_method: {args.score_method}")

    for index, source in enumerate(images,start=1):
        prefix = f"{index:02d}"
        # FFT functions use fixed filenames, so generate in a temporary per-image directory.
        # Only the renamed artifacts are retained in the single final output directory.
        with tempfile.TemporaryDirectory(prefix=f".period_{prefix}_", dir=output_dir) as temp_name:
            work_dir = Path(temp_name)
            previous_argv = sys.argv
            try:
                sys.argv = [
                    "period_candidates_fft2.py",
                    "--image", str(source),
                    "--output_dir", str(work_dir),
                    "--score-method", args.score_method,
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    fft.main()
            finally:
                sys.argv = previous_argv

            generated_files = sorted(work_dir.glob("*.png"))
            if not generated_files:
                raise RuntimeError(f"No outputs generated for input image: {source}")

            for generated in generated_files:
                target = output_dir / output_name(generated.name, index)
                if target.exists():
                    raise FileExistsError(f"Output already exists: {target}")
                # Verify the image before preserving it in the final flat output folder.
                with Image.open(generated) as image:
                    image.verify()
                generated.replace(target)

        print(f"[{index:02d}/{len(images):02d}] {source.name}: saved", flush=True)

    print(f"Completed: {len(images)} input images; all files saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
