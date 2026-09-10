from datetime import datetime
from io import BytesIO

from flask import Blueprint, jsonify, send_file
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from database.models import Project


reports_bp = Blueprint(
    "reports",
    __name__
)


# ============================================================
# REPORT SUMMARY
# ============================================================

def get_report_statistics():

    projects = Project.query.all()

    total = len(projects)

    if total == 0:
        return {
            "total": 0,
            "sanctioned": 0,
            "released": 0,
            "expenditure": 0,
            "completed": 0,
            "ongoing": 0,
            "delayed": 0,
            "anomalies": 0,
            "high_risk": 0,
            "critical": 0,
            "average_risk": 0,
            "financial": 0,
            "duplicates": 0,
            "contractors": 0,
            "geographic": 0,
            "quality": 0,
        }

    sanctioned = sum(
        float(p.sanctioned_amount or 0)
        for p in projects
    )

    released = sum(
        float(p.released_amount or 0)
        for p in projects
    )

    expenditure = sum(
        float(p.expenditure or 0)
        for p in projects
    )

    completed = sum(
        1
        for p in projects
        if str(p.project_status or "").lower()
        in {
            "completed",
            "complete",
            "closed"
        }
    )

    ongoing = sum(
        1
        for p in projects
        if str(p.project_status or "").lower()
        in {
            "ongoing",
            "in progress",
            "under implementation"
        }
    )

    delayed = sum(
        1
        for p in projects
        if float(p.delay_indicator or 0) > 0
    )

    anomalies = sum(
        1
        for p in projects
        if float(p.anomaly_score or 0) > 0
    )

    high_risk = sum(
        1
        for p in projects
        if float(p.risk_score or 0) >= 50
    )

    critical = sum(
        1
        for p in projects
        if float(p.risk_score or 0) >= 75
    )

    financial = sum(
        1
        for p in projects
        if float(p.financial_indicator or 0) > 0
    )

    duplicates = sum(
        1
        for p in projects
        if float(p.duplicate_indicator or 0) > 0
    )

    contractors = sum(
        1
        for p in projects
        if float(p.contractor_indicator or 0) > 0
    )

    geographic = sum(
        1
        for p in projects
        if float(p.geographic_indicator or 0) > 0
    )

    quality = sum(
        1
        for p in projects
        if float(p.data_quality_indicator or 0) > 0
    )

    average_risk = (
        sum(
            float(p.risk_score or 0)
            for p in projects
        )
        / total
    )

    return {
        "total": total,
        "sanctioned": sanctioned,
        "released": released,
        "expenditure": expenditure,
        "completed": completed,
        "ongoing": ongoing,
        "delayed": delayed,
        "anomalies": anomalies,
        "high_risk": high_risk,
        "critical": critical,
        "average_risk": average_risk,
        "financial": financial,
        "duplicates": duplicates,
        "contractors": contractors,
        "geographic": geographic,
        "quality": quality,
    }


# ============================================================
# FORMAT MONEY
# ============================================================

def format_money(value):

    value = float(value or 0)

    if value >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"

    if value >= 100_000:
        return f"₹{value / 100_000:.2f} L"

    return f"₹{value:,.2f}"


# ============================================================
# REPORT API
# ============================================================

@reports_bp.route(
    "/api/reports/summary",
    methods=["GET"]
)
def report_summary():

    return jsonify({
        "success": True,
        "data": get_report_statistics()
    })


# ============================================================
# GENERATE PDF
# ============================================================

@reports_bp.route(
    "/api/reports/pdf",
    methods=["GET"]
)
def generate_pdf():

    stats = get_report_statistics()

    projects = (
        Project.query
        .order_by(
            Project.risk_score.desc()
        )
        .all()
    )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="MPLAD AI Implementation Intelligence Report"
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=19,
        leading=24,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalReport",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=13,
    )

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "MPLAD IMPLEMENTATION INTELLIGENCE REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Powered Anomaly, Risk and Efficiency Assessment",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"Generated: "
            f"{datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            subtitle_style
        )
    )

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "1. Executive Summary",
            heading_style
        )
    )

    summary_text = (
        f"The monitoring system analysed "
        f"<b>{stats['total']:,}</b> projects. "
        f"The total sanctioned amount is "
        f"<b>{format_money(stats['sanctioned'])}</b>, "
        f"with recorded expenditure of "
        f"<b>{format_money(stats['expenditure'])}</b>. "
        f"The system identified "
        f"<b>{stats['anomalies']:,}</b> projects with "
        f"machine-learning anomaly indicators and "
        f"<b>{stats['high_risk']:,}</b> projects classified "
        f"as high or critical risk."
    )

    story.append(
        Paragraph(
            summary_text,
            normal_style
        )
    )

    # ========================================================
    # PROGRAMME STATISTICS
    # ========================================================

    story.append(
        Paragraph(
            "2. Programme Statistics",
            heading_style
        )
    )

    utilization = 0

    if stats["sanctioned"] > 0:
        utilization = (
            stats["expenditure"]
            /
            stats["sanctioned"]
            *
            100
        )

    programme_data = [
        ["Metric", "Value"],
        ["Total Projects", f"{stats['total']:,}"],
        [
            "Sanctioned Amount",
            format_money(stats["sanctioned"])
        ],
        [
            "Released Amount",
            format_money(stats["released"])
        ],
        [
            "Total Expenditure",
            format_money(stats["expenditure"])
        ],
        [
            "Fund Utilization",
            f"{utilization:.2f}%"
        ],
        [
            "Completed Projects",
            f"{stats['completed']:,}"
        ],
        [
            "Ongoing Projects",
            f"{stats['ongoing']:,}"
        ],
        [
            "Delayed Projects",
            f"{stats['delayed']:,}"
        ],
    ]

    table = Table(
        programme_data,
        colWidths=[
            90 * mm,
            80 * mm
        ]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1e3a8a")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#cbd5e1")
            ),
            (
                "FONTNAME",
                (0, 1),
                (-1, -1),
                "Helvetica"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(table)

    # ========================================================
    # AI FINDINGS
    # ========================================================

    story.append(
        Paragraph(
            "3. AI Detection Findings",
            heading_style
        )
    )

    ai_data = [
        ["Indicator", "Projects Flagged"],
        [
            "Machine-Learning Anomalies",
            f"{stats['anomalies']:,}"
        ],
        [
            "Financial Anomalies",
            f"{stats['financial']:,}"
        ],
        [
            "Delayed Projects",
            f"{stats['delayed']:,}"
        ],
        [
            "Duplicate / Similar Patterns",
            f"{stats['duplicates']:,}"
        ],
        [
            "Contractor Concentration",
            f"{stats['contractors']:,}"
        ],
        [
            "Geographic Clusters",
            f"{stats['geographic']:,}"
        ],
        [
            "Data Quality Issues",
            f"{stats['quality']:,}"
        ],
    ]

    ai_table = Table(
        ai_data,
        colWidths=[
            120 * mm,
            50 * mm
        ]
    )

    ai_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1e3a8a")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#cbd5e1")
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(ai_table)

    # ========================================================
    # RISK SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "4. Risk Assessment",
            heading_style
        )
    )

    risk_data = [
        ["Risk Metric", "Value"],
        [
            "Average Risk Score",
            f"{stats['average_risk']:.2f} / 100"
        ],
        [
            "High Risk Projects",
            f"{stats['high_risk']:,}"
        ],
        [
            "Critical Risk Projects",
            f"{stats['critical']:,}"
        ],
    ]

    risk_table = Table(
        risk_data,
        colWidths=[
            120 * mm,
            50 * mm
        ]
    )

    risk_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#991b1b")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#cbd5e1")
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(risk_table)

    # ========================================================
    # HIGH RISK PROJECTS
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "5. Projects Requiring Attention",
            heading_style
        )
    )

    attention_data = [
        [
            "Project ID",
            "Project",
            "District",
            "Risk",
            "Level"
        ]
    ]

    for project in projects[:25]:

        score = float(
            project.risk_score or 0
        )

        if score >= 75:
            level = "CRITICAL"
        elif score >= 50:
            level = "HIGH"
        elif score >= 25:
            level = "MEDIUM"
        else:
            level = "LOW"

        project_name = (
            str(project.project_name or "")
            [:35]
        )

        attention_data.append([
            str(project.project_id or ""),
            project_name,
            str(project.district or ""),
            f"{score:.1f}",
            level
        ])

    attention_table = Table(
        attention_data,
        colWidths=[
            28 * mm,
            70 * mm,
            35 * mm,
            18 * mm,
            25 * mm
        ],
        repeatRows=1
    )

    attention_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1e3a8a")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.HexColor("#cbd5e1")
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                5
            ),
        ])
    )

    story.append(
        attention_table
    )

    # ========================================================
    # METHODOLOGY
    # ========================================================

    story.append(
        Paragraph(
            "6. Methodology",
            heading_style
        )
    )

    methodology = (
        "The system uses Isolation Forest-based unsupervised "
        "machine learning together with rule-based financial "
        "analysis, delay analysis, duplicate/similarity "
        "detection, contractor concentration analysis, "
        "geographic clustering and data-quality validation. "
        "These indicators are combined into a transparent "
        "risk score for decision support."
    )

    story.append(
        Paragraph(
            methodology,
            normal_style
        )
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(
        Paragraph(
            "7. Decision-Support Disclaimer",
            heading_style
        )
    )

    disclaimer = (
        "<b>Important:</b> AI-generated anomaly indicators "
        "and risk scores are not proof of fraud, corruption, "
        "misconduct or wrongdoing. They are intended to assist "
        "authorized officials in prioritising projects for "
        "further examination. Final conclusions should be "
        "based on supporting documents, field verification, "
        "applicable rules and authorised human review."
    )

    story.append(
        Paragraph(
            disclaimer,
            normal_style
        )
    )

    # ========================================================
    # BUILD
    # ========================================================

    document.build(story)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=(
            "MPLAD_AI_Implementation_Report.pdf"
        ),
        mimetype="application/pdf"
    )