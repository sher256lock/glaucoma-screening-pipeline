"""Headless command-line interface for batch and single-image inference."""

from __future__ import annotations

import argparse
import base64
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from src.inference import GlaucomaInferenceEngine, InferenceError
from src.report import generate_clinical_pdf

logger = logging.getLogger(__name__)


def format_terminal_table(results: List[Dict[str, Any]]) -> str:
    """Format inference results into a clear terminal ASCII table."""
    headers = ["Image / File", "Predicted Class", "Glaucoma Suspicion Score", "Threshold", "Latency (ms)", "Quality Warnings"]
    col_widths = [26, 17, 26, 11, 14, 30]

    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_row = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    rows = [header_row, sep_row]
    for r in results:
        fname = str(r.get("file_name", "N/A"))[:col_widths[0]]
        cls_name = str(r.get("predicted_class", "N/A"))[:col_widths[1]]
        score = f"{r.get('suspicion_score', 0.0):.4f}"
        thresh = f"{r.get('threshold_used', 0.0):.4f}"
        lat = f"{r.get('latency_ms', 0.0):.1f}"
        warns = "; ".join(r.get("quality_warnings", []))
        if not warns:
            warns = "None (Pass)"
        warns = warns[:col_widths[5]]

        vals = [fname, cls_name, score, thresh, lat, warns]
        row_str = " | ".join(vals[i].ljust(col_widths[i]) for i in range(len(vals)))
        rows.append(row_str)

    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Glaucoma Screening CLI — Single-image or batch directory inference engine."
    )
    parser.add_argument(
        "--file", "--image", "-f",
        dest="file_path",
        type=str,
        default=None,
        help="Path to a single retinal fundus image (.jpg, .jpeg, .png).",
    )
    parser.add_argument(
        "--dir", "-d",
        dest="dir_path",
        type=str,
        default=None,
        help="Directory containing retinal fundus images for batch screening.",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model_config.yaml configuration file.",
    )
    parser.add_argument(
        "--save-gradcam",
        type=str,
        default=None,
        help="Path to save Grad-CAM overlay image for single-image inference.",
    )
    parser.add_argument(
        "--pdf",
        nargs="?",
        const=True,
        default=False,
        help="Generate and save clinical screening summary PDF report (optional output path).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw machine-readable JSON output instead of ASCII table.",
    )

    args = parser.parse_args()

    if not args.file_path and not args.dir_path:
        parser.print_help()
        sys.exit(1)

    # Initialize inference engine once
    try:
        engine = GlaucomaInferenceEngine(config_path=args.config)
    except Exception as e:
        logger.error(f"Engine initialization failed: {e}")
        sys.exit(1)

    target_files: List[Path] = []
    if args.file_path:
        p = Path(args.file_path)
        if not p.exists() or not p.is_file():
            logger.error(f"Target image file not found: {p}")
            sys.exit(1)
        target_files.append(p)
    elif args.dir_path:
        d = Path(args.dir_path)
        if not d.exists() or not d.is_dir():
            logger.error(f"Target directory not found: {d}")
            sys.exit(1)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            target_files.extend(d.glob(ext))
        target_files = sorted(target_files)
        if not target_files:
            logger.warning(f"No fundus image files found in directory: {d}")
            sys.exit(0)

    results: List[Dict[str, Any]] = []

    for file_path in target_files:
        try:
            with open(file_path, "rb") as f:
                img_bytes = f.read()

            need_cam = bool(args.save_gradcam or args.json or args.pdf)
            res = engine.predict(img_bytes, include_gradcam=need_cam)
            res["file_name"] = file_path.name
            res["file_path"] = str(file_path)

            # Save Grad-CAM overlay if requested
            if args.save_gradcam and res.get("gradcam_overlay_base64"):
                save_out = Path(args.save_gradcam)
                save_out.parent.mkdir(parents=True, exist_ok=True)
                cam_bytes = base64.b64decode(res["gradcam_overlay_base64"])
                with open(save_out, "wb") as f_cam:
                    f_cam.write(cam_bytes)
                logger.info(f"Saved Grad-CAM overlay to: {save_out}")

            # Save clinical PDF report if requested
            if args.pdf:
                if isinstance(args.pdf, str):
                    pdf_out = Path(args.pdf)
                else:
                    pdf_out = Path(f"report_{res['reference_id']}.pdf")
                pdf_out.parent.mkdir(parents=True, exist_ok=True)
                pdf_buf = generate_clinical_pdf(prediction_result=res, original_image=img_bytes)
                with open(pdf_out, "wb") as f_pdf:
                    f_pdf.write(pdf_buf.getvalue())
                res["pdf_path"] = str(pdf_out)
                logger.info(f"Saved clinical PDF report to: {pdf_out}")

            results.append(res)
        except InferenceError as e:
            logger.error(f"Inference failed for {file_path.name}: {e}")
            results.append({
                "file_name": file_path.name,
                "file_path": str(file_path),
                "error": str(e),
                "predicted_class": "ERROR",
                "suspicion_score": 0.0,
                "threshold_used": engine.threshold,
                "latency_ms": 0.0,
                "quality_warnings": [str(e)],
            })

    if args.json:
        # Avoid dumping huge base64 strings if not strictly needed in table output
        clean_results = []
        for r in results:
            item = dict(r)
            if not args.save_gradcam and len(item.get("gradcam_overlay_base64", "")) > 100:
                item["gradcam_overlay_base64"] = f"<base64_png_len_{len(item['gradcam_overlay_base64'])}>"
                item["gradcam_base64"] = item["gradcam_overlay_base64"]
            clean_results.append(item)
        print(json.dumps(clean_results if len(clean_results) > 1 else clean_results[0], indent=2))
    else:
        print("\n" + "=" * 115)
        print("GLAUCOMA SCREENING INFERENCE REPORT")
        print("=" * 115)
        print(format_terminal_table(results))
        print("=" * 115)
        print(f"Decision Threshold: {engine.threshold:.4f} ({engine.threshold_procedure})")
        print(f"Regulatory Disclaimer: {engine.disclaimer}")
        print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
