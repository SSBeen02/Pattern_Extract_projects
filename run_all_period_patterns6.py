"""Run FFT6 on every input image and collect numbered outputs in one folder."""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile

from PIL import Image

import period_candidates_fft6 as fft
from run_all_period_patterns5 import IMAGE_SUFFIXES, natural_sort_key, output_name


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run FFT6's half-NCC, half-VGG19-conv4_4 ranking on all images."
    )
    parser.add_argument("--input-dir", "--input_dir", type=Path,
                        default=Path("aligned_1024ver"))
    parser.add_argument("--output-dir", "--output_dir", type=Path,
                        default=Path("outputs/fft6_all"))
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
                    "period_candidates_fft6.py", "--image", str(source),
                    "--output_dir", str(work_dir),
                    "--fout-opt", args.fout_opt,
                    "--pout-opt", args.pout_opt,
                ]
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    fft.main()
            finally:
                sys.argv = previous_argv

            generated_files = sorted(path for path in work_dir.iterdir()
                                     if path.suffix.lower() in (".png", ".csv"))
            if not generated_files:
                raise RuntimeError(f"No outputs generated for input image: {source}")
            targets = [output_dir / output_name(path.name, index) for path in generated_files]
            existing = next((target for target in targets if target.exists()), None)
            if existing is not None:
                raise FileExistsError(f"Output already exists: {existing}")
            for generated, target in zip(generated_files, targets):
                if generated.suffix.lower() == ".png":
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
