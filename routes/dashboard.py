from flask import Blueprint, jsonify
from sqlalchemy import func

from database.db import db
from database.models import Project


dashboard_bp = Blueprint(
    "dashboard",
    __name__
)


@dashboard_bp.route(
    "/api/dashboard",
    methods=["GET"]
)
def dashboard_api():

    # --------------------------------------------------
    # TOTAL PROJECTS
    # --------------------------------------------------

    total_projects = Project.query.count()

    # --------------------------------------------------
    # TOTAL SANCTIONED AMOUNT
    # --------------------------------------------------

    total_sanctioned = (
        db.session.query(
            func.coalesce(
                func.sum(
                    Project.sanctioned_amount
                ),
                0
            )
        ).scalar()
        or 0
    )

    # --------------------------------------------------
    # TOTAL EXPENDITURE
    # --------------------------------------------------

    total_expenditure = (
        db.session.query(
            func.coalesce(
                func.sum(
                    Project.expenditure
                ),
                0
            )
        ).scalar()
        or 0
    )

    # --------------------------------------------------
    # FUND UTILIZATION
    # --------------------------------------------------

    utilization = 0

    if total_sanctioned > 0:

        utilization = (
            total_expenditure /
            total_sanctioned
        ) * 100

    # --------------------------------------------------
    # COMPLETED PROJECTS
    # --------------------------------------------------

    completed = Project.query.filter(
        func.lower(
            Project.project_status
        ).in_([
            "completed",
            "complete"
        ])
    ).count()

    # --------------------------------------------------
    # ONGOING PROJECTS
    # --------------------------------------------------

    ongoing = Project.query.filter(
        func.lower(
            Project.project_status
        ).in_([
            "ongoing",
            "in progress",
            "under implementation"
        ])
    ).count()

    # --------------------------------------------------
    # DELAYED PROJECTS
    # --------------------------------------------------

    delayed = Project.query.filter(
        func.lower(
            Project.project_status
        ).in_([
            "delayed",
            "delay"
        ])
    ).count()

    # --------------------------------------------------
    # HIGH-RISK PROJECTS
    # --------------------------------------------------

    high_risk = Project.query.filter(
        Project.risk_level.in_([
            "HIGH",
            "CRITICAL"
        ])
    ).count()

    # --------------------------------------------------
    # CRITICAL PROJECTS
    # --------------------------------------------------

    critical = Project.query.filter(
        Project.risk_level == "CRITICAL"
    ).count()

    # --------------------------------------------------
    # DETECTED ANOMALIES
    # --------------------------------------------------

    anomalies = Project.query.filter(
        Project.anomaly_score > 0
    ).count()

    # --------------------------------------------------
    # RESPONSE
    # --------------------------------------------------

    return jsonify({

        "success": True,

        "data": {

            "total_projects":
                total_projects,

            "total_sanctioned_amount":
                round(
                    total_sanctioned,
                    2
                ),

            "total_expenditure":
                round(
                    total_expenditure,
                    2
                ),

            "funds_utilized_percent":
                round(
                    utilization,
                    2
                ),

            "completed_projects":
                completed,

            "ongoing_projects":
                ongoing,

            "delayed_projects":
                delayed,

            "high_risk_projects":
                high_risk,

            "critical_projects":
                critical,

            "detected_anomalies":
                anomalies
        }
    })