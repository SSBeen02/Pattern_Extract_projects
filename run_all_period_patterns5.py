"""Run FFT5 on every input pattern and collect numbered outputs in one folder."""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import re
import sys
import tempfile

from PIL import Image

import period_candidates_fft5 as fft


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_sort_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.as_posix())]


def output_name(generated_name: str, index: int) -> str:
    if generated_name.startswith(f"{index:02d}_"):
        return generated_name
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
        description="Run FFT5's half-dense, half-conv3_4 ranking on all input patterns."
    )
    parser.add_argument("--input-dir", "--input_dir", type=Path,
                        default=Path("aligned_1024ver"),
                        help="Input image folder, searched recursively (default: aligned_1024ver).")
    parser.add_argument("--output-dir", "--output_dir", type=Path,
                        default=Path("outputs/fft5_all"),
                        help="Flat output folder (default: outputs/fft5_all).")
    parser.add_argument("--fout-opt", choices=("on", "off"), default="on")
    parser.add_argument("--pout-opt", choices=("on", "off"), default="on")
    args = parser.parse_args()

    input_dir = (args.input_dir if args.input_dir.is_absolute() else root / args.input_dir).resolve()
    output_dir = (args.output_dir if args.output_dir.is_absolute() else root / args.output_dir).resolve()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        parser.error("Output folder must be outside the input folder.")
    images = sorted(
        (path for path in input_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_sort_key,
    )
    if not images:
        parser.error(f"No input images found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    for index, source in enumerate(images, start=1):
        with tempfile.TemporaryDirectory(prefix=f".period_{index:02d}_", dir=output_dir) as temp_name:
            work_dir = Path(temp_name)
            previous_argv = sys.argv
            try:
                sys.argv = [
                    "period_candidates_fft5.py", "--image", str(source),
                    "--output_dir", str(work_dir),
                    "--fout-opt", args.fout_opt,
                    "--pout-opt", args.pout_opt,
                ]
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    fft.main()
            finally:
                sys.argv = previous_argv

            generated_files = sorted(work_dir.glob("*.png"))
            if not generated_files:
                raise RuntimeError(f"No outputs generated for input image: {source}")
            targets = [output_dir / output_name(file.name, index) for file in generated_files]
            existing = next((target for target in targets if target.exists()), None)
            if existing is not None:
                raise FileExistsError(f"Output already exists: {existing}")
            for generated, target in zip(generated_files, targets):
                with Image.open(generated) as image:
                    image.verify()
                with target.open("xb") as handle:
                    handle.write(generated.read_bytes())

        rank_one = next((line for line in captured.getvalue().splitlines()
                         if line.startswith("rank=1 ")), "")
        print(f"[{index:02d}/{len(images):02d}] {source.name}: saved {rank_one}", flush=True)
    print(f"Completed: {len(images)} images; outputs in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
