"""Single-page clinical screening summary PDF generator using ReportLab in-memory BytesIO."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import io
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _prepare_image_buffer(
    img_input: Union[bytes, str, Path, Image.Image, np.ndarray],
) -> io.BytesIO:
    """Normalize various image formats into an in-memory PNG BytesIO buffer for ReportLab."""
    buf = io.BytesIO()

    if isinstance(img_input, (str, Path)):
        p = Path(img_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        with Image.open(p) as pil_img:
            pil_img.convert("RGB").save(buf, format="PNG")
    elif isinstance(img_input, bytes):
        with Image.open(io.BytesIO(img_input)) as pil_img:
            pil_img.convert("RGB").save(buf, format="PNG")
    elif isinstance(img_input, Image.Image):
        img_input.convert("RGB").save(buf, format="PNG")
    elif isinstance(img_input, np.ndarray):
        if img_input.dtype != np.uint8:
            img_input = np.clip(img_input * 255.0, 0, 255).astype(np.uint8)
        if img_input.ndim == 2:
            Image.fromarray(img_input).convert("RGB").save(buf, format="PNG")
        else:
            Image.fromarray(img_input).save(buf, format="PNG")
    else:
        raise ValueError(f"Unsupported image input type: {type(img_input)}")

    buf.seek(0)
    return buf


def generate_clinical_pdf(
    prediction_result: Dict[str, Any],
    original_image: Union[bytes, str, Path, Image.Image, np.ndarray],
    gradcam_image: Optional[Union[bytes, str, Image.Image, np.ndarray]] = None,
) -> io.BytesIO:
    """Generate a single-page clinical screening summary PDF in-memory.

    Strictly complies with healthcare PHI regulations:
    - Contains zero patient names, DOBs, medical record numbers, or PHI.
    - Uses UUID request tracking and UTC generation timestamp.
    - Strictly employs "Glaucoma Suspicion Score" terminology.
    - Includes verbatim Research Prototype Disclaimer.
    - Never writes PDF bytes to disk (pure in-memory BytesIO stream).

    Args:
        prediction_result: Output dictionary from GlaucomaInferenceEngine.predict().
        original_image: Input fundus photograph (bytes, PIL, path, or numpy).
        gradcam_image: Optional override for Grad-CAM overlay image. If None,
            extracts from prediction_result['gradcam_overlay_base64'].

    Returns:
        io.BytesIO buffer positioned at offset 0 containing valid PDF bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=10,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10,
        alignment=1,  # Center
        textColor=colors.HexColor("#64748B"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # 1. Header Banner & Title
    story.append(Paragraph("Glaucoma Screening Prototype Report", title_style))
    story.append(
        Paragraph("ResNet-18 Deep Learning Binary Decision Support &bull; REFUGE Protocol", subtitle_style)
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=10))

    # 2. Metadata & De-Identification Summary
    ref_id = prediction_result.get("reference_id", "req_unknown")
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    model_version = prediction_result.get("model_version", "resnet18-refuge-v1")

    meta_table_data = [
        [
            Paragraph(f"<b>Reference ID:</b> <font name='Courier'>{ref_id}</font>", body_style),
            Paragraph(f"<b>Screening Timestamp:</b> {timestamp_utc}", body_style),
        ],
        [
            Paragraph(f"<b>Model Architecture:</b> {model_version}", body_style),
            Paragraph("<b>PHI Status:</b> <font color='#059669'>De-Identified (Zero PHI)</font>", body_style),
        ],
    ]
    meta_table = Table(meta_table_data, colWidths=[270, 270])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Diagnostic Screening Findings Table
    pred_class = str(prediction_result.get("predicted_class", "Normal"))
    suspicion_score = float(prediction_result.get("suspicion_score", 0.0))
    threshold_used = float(prediction_result.get("threshold_used", 0.2921))
    threshold_proc = str(prediction_result.get("threshold_procedure", "youden_j_on_val"))
    latency_ms = float(prediction_result.get("latency_ms", 0.0))

    is_glaucoma = (pred_class == "Glaucoma")
    outcome_color = "#DC2626" if is_glaucoma else "#059669"
    outcome_label = "Glaucoma Suspicion Flagged" if is_glaucoma else "Normal / Non-Glaucoma"

    diag_data = [
        [
            Paragraph("<b>Predicted Classification</b>", body_style),
            Paragraph(f"<b><font color='{outcome_color}' size='11'>{outcome_label}</font></b>", body_style),
        ],
        [
            Paragraph("<b>Glaucoma Suspicion Score</b>", body_style),
            Paragraph(f"<b><font size='11' name='Courier'>{suspicion_score:.4f}</font></b> (Scale: 0.0 to 1.0)", body_style),
        ],
        [
            Paragraph("<b>Decision Threshold</b>", body_style),
            Paragraph(f"<font name='Courier'>{threshold_used:.4f}</font> ({threshold_proc})", body_style),
        ],
        [
            Paragraph("<b>In-Memory CPU Latency</b>", body_style),
            Paragraph(f"{latency_ms:.1f} ms", body_style),
        ],
    ]
    diag_table = Table(diag_data, colWidths=[200, 340])
    diag_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(Paragraph("Diagnostic Screening Summary", section_heading))
    story.append(diag_table)
    story.append(Spacer(1, 10))

    # 4. Side-by-Side Fundus & Grad-CAM Images
    story.append(Paragraph("Visual Evidence & Explainability Attribution", section_heading))

    orig_buf = _prepare_image_buffer(original_image)

    # Prepare GradCAM image buffer
    if gradcam_image is not None:
        cam_buf = _prepare_image_buffer(gradcam_image)
    else:
        b64_str = prediction_result.get("gradcam_overlay_base64", "") or prediction_result.get("gradcam_base64", "")
        if b64_str:
            cam_bytes = base64.b64decode(b64_str)
            cam_buf = _prepare_image_buffer(cam_bytes)
        else:
            # Fallback placeholder if GradCAM not available
            dummy_img = Image.new("RGB", (224, 224), color=(30, 40, 60))
            cam_buf = _prepare_image_buffer(dummy_img)

    img_dim = 195  # points (~2.7 inches)
    rl_orig = RLImage(orig_buf, width=img_dim, height=img_dim)
    rl_cam = RLImage(cam_buf, width=img_dim, height=img_dim)

    img_table_data = [
        [rl_orig, rl_cam],
        [
            Paragraph("<b>Original Digital Fundus Photograph</b>", caption_style),
            Paragraph("<b>Grad-CAM Attribution Overlay (layer4[-1])</b>", caption_style),
        ],
    ]
    img_table = Table(img_table_data, colWidths=[270, 270])
    img_table.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    story.append(img_table)
    story.append(Spacer(1, 10))

    # 5. Quality Warnings Heuristics Box
    warnings = prediction_result.get("quality_warnings", [])
    story.append(Paragraph("Image Quality Assessment", section_heading))
    if warnings:
        warn_text = "<br/>".join([f"&bull; <font color='#B45309'>{w}</font>" for w in warnings])
        warn_p = Paragraph(f"<b>Quality Heuristic Warnings Flagged:</b><br/>{warn_text}", body_style)
        warn_box = Table([[warn_p]], colWidths=[540])
        warn_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#F59E0B")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(warn_box)
    else:
        pass_p = Paragraph(
            "<b>Quality Heuristics:</b> <font color='#059669'>All checks passed.</font> "
            "Laplacian sharpness variance and mean luminance are within operational screening specifications.",
            body_style,
        )
        pass_box = Table([[pass_p]], colWidths=[540])
        pass_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#86EFAC")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(pass_box)

    story.append(Spacer(1, 12))

    # 6. Verbatim Regulatory Disclaimer Box
    disclaimer_html = (
        "<b>REGULATORY &amp; RESEARCH PROTOTYPE DISCLAIMER:</b><br/>"
        "RESEARCH PROTOTYPE ONLY. Glaucoma Suspicion Score is not a calibrated probability and is not for clinical diagnosis.<br/>"
        "Model outputs are reported strictly as a <b>'Glaucoma Suspicion Score'</b> derived from a class-weighted binary cross-entropy "
        "objective (BCEWithLogitsLoss). Because class weighting shifts the prior log-odds, these scores are not calibrated statistical "
        "probabilities and must never be presented as 'confidence' or 'probability.' All screening outputs require formal diagnostic "
        "evaluation and clinical confirmation by a qualified ophthalmologist."
    )
    disclaimer_p = Paragraph(disclaimer_html, disclaimer_style)
    disclaimer_table = Table([[disclaimer_p]], colWidths=[540])
    disclaimer_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(disclaimer_table)

    # Build PDF into in-memory buffer
    doc.build(story)
    buffer.seek(0)
    return buffer
