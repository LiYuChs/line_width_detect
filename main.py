import argparse
from datetime import datetime
from pathlib import Path

from fhwm_main import FWHMProcessor
from optimized_sam_main import OptimizedSAMProcessor
from SAM_main import SAMProcessor
from sr_sam_main import SRSAMProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Line width measurement runner")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["fwhm", "sam", "optimized_sam", "sr_sam"],
        help="Measurement method to run",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./data/measure_01.jpg",
        help="Input image path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (optional)",
    )
    return parser.parse_args()


def build_default_output_dir(method):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "data" / f"{method}_result" / timestamp


def run_selected_method(method, image_path, output_dir=None):
    image_path = Path(image_path)
    output_dir = Path(output_dir) if output_dir else build_default_output_dir(method)

    if method == "fwhm":
        processor = FWHMProcessor()
        return processor.run(image_path=image_path, output_dir=output_dir)

    if method == "sam":
        processor = SAMProcessor()
        return processor.run(image_path=image_path, output_dir=output_dir)

    if method == "optimized_sam":
        processor = OptimizedSAMProcessor()
        return processor.run(image_path=image_path, output_dir=output_dir)

    if method == "sr_sam":
        processor = SRSAMProcessor()
        return processor.run(image_path=image_path, output_dir=output_dir)

    raise ValueError(f"Unsupported method: {method}")


def main():
    args = parse_args()
    run_result = run_selected_method(args.method, args.image, args.output_dir)

    print("=" * 50)
    print(f"Method: {run_result.get('method')}")
    print(f"Image: {run_result.get('image_path')}")
    print(f"Output directory: {run_result.get('output_dir')}")

    for key in ["output_path", "sam_output_path", "morph_output_path", "final_overlay_path", "report_path"]:
        if key in run_result:
            print(f"{key}: {run_result[key]}")


if __name__ == "__main__":
    main()
