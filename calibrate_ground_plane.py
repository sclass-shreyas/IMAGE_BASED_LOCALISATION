"""Create a ground-plane homography from measured image/ground point pairs."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points_json", help="JSON containing image size and >=4 point pairs")
    parser.add_argument("output_json", help="where to write the calibration")
    parser.add_argument("--max-residual-m", type=float, default=0.15)
    args = parser.parse_args()
    from src.ground_calibration import fit_calibration
    source = json.loads(Path(args.points_json).read_text(encoding="utf-8"))
    try:
        result = fit_calibration(source, args.max_residual_m)
    except ValueError as exc:
        raise SystemExit(f"Calibration rejected: {exc}") from exc
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Held-out residuals (m):", result["validation_residuals_m"] or "none supplied")
    print(f"Wrote {args.output_json}; calibration point RMS error: {result['rms_error_m']:.3f} m")


if __name__ == "__main__":
    main()
