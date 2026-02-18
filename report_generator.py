# src/report_generator.py
"""
Professional multi-section PDF report generator for NIDS sessions (Option A).
- Cover page (no prepared-by info)
- Executive summary
- Attack count bar chart
- Protocol distribution pie chart
- Paginated detailed packet table (with severity color coding)
- Returns the path of the generated PDF

Requires:
    pip install reportlab matplotlib
"""

import os
import json
import tempfile
from datetime import datetime
from math import ceil

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Matplotlib for charts
import matplotlib.pyplot as plt

# threat mapping import (re-use your mapping)
try:
    from src.threat_mapping import map_prediction_to_threat
except Exception:
    # fallback minimal mapping
    def map_prediction_to_threat(pred):
        return {"category": "Unknown", "description": "", "cves": [], "cvss": None, "severity": "Unknown"}


# Severity -> color mapping for table cell background (light)
SEVERITY_COLORS = {
    "critical": colors.HexColor("#D32F2F"),  # red
    "high": colors.HexColor("#E53935"),
    "medium": colors.HexColor("#FB8C00"),  # orange
    "low": colors.HexColor("#FBC02D"),    # yellow
    "info": colors.HexColor("#2E7D32"),   # green
    "unknown": colors.HexColor("#9E9E9E"),
}

# helper to normalize severity
def normalize_severity(s):
    if s is None:
        return "unknown"
    s = str(s).strip().lower()
    if "crit" in s:
        return "critical"
    if s in ("high",):
        return "high"
    if "med" in s or s == "medium":
        return "medium"
    if "low" in s:
        return "low"
    if s in ("info", "informational"):
        return "info"
    return "unknown"


def _shorten(text, n=60):
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= n else s[: n - 3] + "..."


def _build_charts(packets, tmpdir):
    """
    Produce two charts:
    - bar chart: attack_count vs benign (and severity breakdown)
    - pie chart: protocol distribution
    Writes PNG files into tmpdir and returns their file paths.
    """
    # Attack counts and severity breakdown
    total = len(packets)
    attack_total = 0
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0, "unknown": 0}
    proto_counts = {}

    for p in packets:
        pred = str(p.get("prediction", "")).strip().lower()
        mapped = p.get("threat_mapping") or map_prediction_to_threat(p.get("prediction"))
        sev = normalize_severity(mapped.get("severity"))
        if pred in ("attack", "malicious", "1", "true"):
            attack_total += 1
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        else:
            # treat as benign
            severity_counts["info"] = severity_counts.get("info", 0)  # keep
        # protocol
        proto = str(p.get("protocol", "OTHER")).upper()
        proto_counts[proto] = proto_counts.get(proto, 0) + 1

    # --- Chart 1: Attack severity bar chart (stacked) ---
    # We'll create a simple bar showing counts per severity (only non-zero)
    sev_labels = []
    sev_values = []
    sev_colors = []
    for sev in ["critical", "high", "medium", "low", "info", "unknown"]:
        cnt = severity_counts.get(sev, 0)
        if cnt > 0:
            sev_labels.append(sev.capitalize())
            sev_values.append(cnt)
            sev_colors.append(SEVERITY_COLORS.get(sev, colors.grey).hexval()[1:])  # hex string without '#'

    bar_png = None
    pie_png = None

    # generate bar if any attacks exist
    if sum(sev_values) > 0:
        plt.figure(figsize=(6, 3.5))
        bars = plt.bar(sev_labels, sev_values, color=["#" + c for c in sev_colors])
        plt.title("Attack Count by Severity")
        plt.ylabel("Number of Packets")
        for rect in bars:
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, h + 0.5, f"{int(h)}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        bar_png = os.path.join(tmpdir, "attack_severity_bar.png")
        plt.savefig(bar_png, dpi=150)
        plt.close()

    # --- Chart 2: Protocol distribution pie chart ---
    # Only include protocols with >0
    proto_labels = []
    proto_values = []
    for k, v in sorted(proto_counts.items(), key=lambda x: -x[1]):
        proto_labels.append(k)
        proto_values.append(v)

    if len(proto_values) > 0:
        plt.figure(figsize=(6, 4))
        # colors: use matplotlib default
        patches, texts, autotexts = plt.pie(proto_values, labels=proto_labels, autopct="%1.1f%%", startangle=140)
        plt.title("Protocol Distribution")
        plt.axis("equal")
        plt.tight_layout()
        pie_png = os.path.join(tmpdir, "protocol_pie.png")
        plt.savefig(pie_png, dpi=150)
        plt.close()

    return bar_png, pie_png


def generate_pdf_report(packets, out_path):
    """
    Generate professional multi-section PDF:
    - packets: list of dicts (newest-first or any order). Each packet may optionally have 'threat_mapping'.
    - out_path: absolute path where PDF will be written.

    Returns out_path on success; raises on failure.
    """
    # ensure out dir exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Ensure threat mapping present on each packet (augment)
    for i, p in enumerate(packets):
        if "threat_mapping" not in p or not p.get("threat_mapping"):
            p["threat_mapping"] = map_prediction_to_threat(p.get("prediction"))

    # Summary stats
    total_pkts = len(packets)
    attack_pkts = 0
    benign_pkts = 0
    protocol_counts = {}
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0, "unknown": 0}

    for p in packets:
        pred = str(p.get("prediction", "")).strip().lower()
        if pred in ("attack", "malicious", "1", "true"):
            attack_pkts += 1
        else:
            benign_pkts += 1
        proto = str(p.get("protocol", "OTHER")).upper()
        protocol_counts[proto] = protocol_counts.get(proto, 0) + 1
        sev = normalize_severity(p.get("threat_mapping", {}).get("severity"))
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Create temporary directory for charts
    tmpdir = tempfile.mkdtemp(prefix="nids_report_")
    try:
        bar_png, pie_png = _build_charts(packets, tmpdir)

        # build PDF
        doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
        styles = getSampleStyleSheet()
        styleH = ParagraphStyle("Title", parent=styles["Heading1"], alignment=TA_CENTER, fontSize=18, spaceAfter=6)
        styleN = ParagraphStyle("NormalLeft", parent=styles["Normal"], alignment=TA_LEFT, fontSize=9)
        styleBold = ParagraphStyle("Bold", parent=styles["Normal"], fontSize=10, leading=12)
        styleRight = ParagraphStyle("Right", parent=styles["Normal"], alignment=TA_RIGHT, fontSize=9)

        elems = []

        # --- Cover Page ---
        elems.append(Spacer(1, 30))
        elems.append(Paragraph("Network Intrusion Detection System (NIDS) — Session Report", styleH))
        elems.append(Spacer(1, 12))
        elems.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styleRight))
        elems.append(Spacer(1, 12))
        # Session quick stats
        summary_paragraph = f"<b>Total packets:</b> {total_pkts} &nbsp;&nbsp; <b>Attacks:</b> {attack_pkts} &nbsp;&nbsp; <b>Benign:</b> {benign_pkts}"
        elems.append(Paragraph(summary_paragraph, styleN))
        elems.append(Spacer(1, 20))

        # brief note
        elems.append(Paragraph("This report contains protocol and threat analysis based on in-line model predictions and threat-mapping.", styleN))
        elems.append(PageBreak())

        # --- Executive Summary ---
        elems.append(Paragraph("Executive Summary", styles["Heading2"]))
        elems.append(Spacer(1, 6))
        exec_lines = []
        exec_lines.append(Paragraph(f"<b>Total packets captured:</b> {total_pkts}", styleN))
        exec_lines.append(Paragraph(f"<b>Packets detected as attacks:</b> {attack_pkts}", styleN))
        exec_lines.append(Paragraph(f"<b>Packets detected as benign:</b> {benign_pkts}", styleN))
        # Most common protocols
        proto_sorted = sorted(protocol_counts.items(), key=lambda x: -x[1])
        if proto_sorted:
            top_protos = ", ".join([f"{k} ({v})" for k, v in proto_sorted[:5]])
            exec_lines.append(Paragraph(f"<b>Top protocols:</b> {top_protos}", styleN))
        # severity breakdown
        sev_str = ", ".join([f"{k.capitalize()}: {v}" for k, v in severity_counts.items() if v > 0])
        exec_lines.append(Paragraph(f"<b>Severity breakdown:</b> {sev_str}", styleN))
        elems.extend(exec_lines)
        elems.append(PageBreak())

        # --- Charts ---
        elems.append(Paragraph("Charts", styles["Heading2"]))
        elems.append(Spacer(1, 6))

        if bar_png:
            elems.append(Paragraph("Attack Count by Severity", styleBold))
            elems.append(Spacer(1, 6))
            elems.append(Image(bar_png, width=450, height=180))
            elems.append(Spacer(1, 12))

        if pie_png:
            elems.append(Paragraph("Protocol Distribution", styleBold))
            elems.append(Spacer(1, 6))
            elems.append(Image(pie_png, width=450, height=180))
            elems.append(Spacer(1, 12))

        elems.append(PageBreak())

        # --- Detailed Packet Table ---
        elems.append(Paragraph("Detailed Packet Listing", styles["Heading2"]))
        elems.append(Spacer(1, 6))

        # Table headers
        headers = [
            "Timestamp",
            "Src IP",
            "Dst IP",
            "Proto",
            "SrcPort",
            "DstPort",
            "Prediction",
            "Confidence",
            "Model",
            "Severity",
            "CVSS",
            "CVEs",
            "Threat Category",
        ]
        data = [headers]

        # Append rows (limit field lengths)
        for p in packets:
            tm = _shorten(p.get("timestamp", ""))
            s_ip = _shorten(p.get("src_ip", ""))
            d_ip = _shorten(p.get("dst_ip", ""))
            proto = _shorten(p.get("protocol", ""))
            ps = _shorten(p.get("port_src", ""))
            pd = _shorten(p.get("port_dst", ""))
            pred = _shorten(p.get("prediction", ""))
            conf = _shorten(p.get("confidence", ""))
            model = _shorten(p.get("model_used", ""))
            mapping = p.get("threat_mapping") or map_prediction_to_threat(p.get("prediction"))
            severity = mapping.get("severity", "Unknown")
            severity_norm = normalize_severity(severity)
            cvss = mapping.get("cvss", "")
            cves = mapping.get("cves") or []
            cves_str = ", ".join(cves) if isinstance(cves, (list, tuple)) else str(cves)
            category = mapping.get("category", "")

            row = [tm, s_ip, d_ip, proto, ps, pd, pred, conf, model, severity_norm.capitalize(), str(cvss or ""), cves_str, category]
            data.append(row)

        # Build table with style: we will color severity cells
        tbl = Table(data, repeatRows=1, colWidths=[
            70, 70, 70, 50, 40, 40, 60, 50, 50, 50, 40, 120, 80
        ])
        tbl_style = TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E3B4E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
        # apply severity color backgrounds per row (find their column index)
        sev_col_idx = headers.index("Severity")
        for ridx, row in enumerate(data[1:], start=1):
            sev_text = str(row[sev_col_idx]).strip().lower()
            color_key = normalize_severity(sev_text)
            color = SEVERITY_COLORS.get(color_key, colors.HexColor("#FFFFFF"))
            # set a light background for the severity cell
            tbl_style.add("BACKGROUND", (sev_col_idx, ridx), (sev_col_idx, ridx), color)
            # set text color to white for dark backgrounds (critical/high)
            if color_key in ("critical", "high"):
                tbl_style.add("TEXTCOLOR", (sev_col_idx, ridx), (sev_col_idx, ridx), colors.white)
            else:
                tbl_style.add("TEXTCOLOR", (sev_col_idx, ridx), (sev_col_idx, ridx), colors.black)

        tbl.setStyle(tbl_style)

        elems.append(tbl)

        # Optionally a final page break
        # elems.append(PageBreak())

        # Build PDF
        doc.build(elems)
    finally:
        # cleanup temporary chart files & directory
        try:
            for f in os.listdir(tmpdir):
                try:
                    os.remove(os.path.join(tmpdir, f))
                except Exception:
                    pass
            try:
                os.rmdir(tmpdir)
            except Exception:
                pass
        except Exception:
            pass

    return out_path
