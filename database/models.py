from datetime import datetime

from database.db import db


class Project(db.Model):

    __tablename__ = "projects"

    # --------------------------------------------------
    # BASIC PROJECT INFORMATION
    # --------------------------------------------------

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    project_id = db.Column(
        db.String(100),
        unique=True,
        nullable=False,
        index=True
    )

    state = db.Column(
        db.String(100)
    )

    district = db.Column(
        db.String(100)
    )

    constituency = db.Column(
        db.String(150)
    )

    mp_name = db.Column(
        db.String(150)
    )

    project_name = db.Column(
        db.String(500)
    )

    project_category = db.Column(
        db.String(150)
    )

    # --------------------------------------------------
    # FINANCIAL INFORMATION
    # --------------------------------------------------

    sanctioned_amount = db.Column(
        db.Float,
        default=0.0
    )

    released_amount = db.Column(
        db.Float,
        default=0.0
    )

    expenditure = db.Column(
        db.Float,
        default=0.0
    )

    # --------------------------------------------------
    # PROJECT STATUS
    # --------------------------------------------------

    project_status = db.Column(
        db.String(100)
    )

    # --------------------------------------------------
    # TIMELINE
    # --------------------------------------------------

    sanction_date = db.Column(
        db.Date
    )

    start_date = db.Column(
        db.Date
    )

    completion_date = db.Column(
        db.Date
    )

    # --------------------------------------------------
    # IMPLEMENTATION
    # --------------------------------------------------

    implementing_agency = db.Column(
        db.String(200)
    )

    contractor = db.Column(
        db.String(200)
    )

    # --------------------------------------------------
    # GEOGRAPHIC INFORMATION
    # --------------------------------------------------

    latitude = db.Column(
        db.Float
    )

    longitude = db.Column(
        db.Float
    )

    # --------------------------------------------------
    # BENEFICIARIES
    # --------------------------------------------------

    beneficiary_count = db.Column(
        db.Integer
    )

    # --------------------------------------------------
    # AI ANOMALY DETECTION
    # --------------------------------------------------

    anomaly_score = db.Column(
        db.Float,
        default=0.0
    )

    # --------------------------------------------------
    # OVERALL RISK
    # --------------------------------------------------

    risk_score = db.Column(
        db.Float,
        default=0.0
    )

    risk_level = db.Column(
        db.String(20),
        default="LOW"
    )

    # --------------------------------------------------
    # INDIVIDUAL RISK INDICATORS
    # --------------------------------------------------

    financial_indicator = db.Column(
        db.Float,
        default=0.0
    )

    delay_indicator = db.Column(
        db.Float,
        default=0.0
    )

    duplicate_indicator = db.Column(
        db.Float,
        default=0.0
    )

    contractor_indicator = db.Column(
        db.Float,
        default=0.0
    )

    geographic_indicator = db.Column(
        db.Float,
        default=0.0
    )

    data_quality_indicator = db.Column(
        db.Float,
        default=0.0
    )

    # --------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------

    anomaly_reasons = db.Column(
        db.Text,
        default=""
    )

    data_quality_warnings = db.Column(
        db.Text,
        default=""
    )

    # --------------------------------------------------
    # RECORD TIMESTAMPS
    # --------------------------------------------------

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # --------------------------------------------------
    # CONVERT DATABASE OBJECT TO DICTIONARY
    # --------------------------------------------------

    def to_dict(self):

        return {

            "id":
                self.id,

            "project_id":
                self.project_id,

            "state":
                self.state,

            "district":
                self.district,

            "constituency":
                self.constituency,

            "mp_name":
                self.mp_name,

            "project_name":
                self.project_name,

            "project_category":
                self.project_category,

            # Financial
            "sanctioned_amount":
                self.sanctioned_amount,

            "released_amount":
                self.released_amount,

            "expenditure":
                self.expenditure,

            # Status
            "project_status":
                self.project_status,

            # Dates
            "sanction_date":
                (
                    self.sanction_date.isoformat()
                    if self.sanction_date
                    else None
                ),

            "start_date":
                (
                    self.start_date.isoformat()
                    if self.start_date
                    else None
                ),

            "completion_date":
                (
                    self.completion_date.isoformat()
                    if self.completion_date
                    else None
                ),

            # Implementation
            "implementing_agency":
                self.implementing_agency,

            "contractor":
                self.contractor,

            # Geography
            "latitude":
                self.latitude,

            "longitude":
                self.longitude,

            # Beneficiaries
            "beneficiary_count":
                self.beneficiary_count,

            # AI
            "anomaly_score":
                round(
                    self.anomaly_score or 0,
                    2
                ),

            "risk_score":
                round(
                    self.risk_score or 0,
                    2
                ),

            "risk_level":
                self.risk_level,

            # Indicators
            "financial_indicator":
                round(
                    self.financial_indicator or 0,
                    2
                ),

            "delay_indicator":
                round(
                    self.delay_indicator or 0,
                    2
                ),

            "duplicate_indicator":
                round(
                    self.duplicate_indicator or 0,
                    2
                ),

            "contractor_indicator":
                round(
                    self.contractor_indicator or 0,
                    2
                ),

            "geographic_indicator":
                round(
                    self.geographic_indicator or 0,
                    2
                ),

            "data_quality_indicator":
                round(
                    self.data_quality_indicator or 0,
                    2
                ),

            # Explanations
            "anomaly_reasons":
                (
                    self.anomaly_reasons.split("||")
                    if self.anomaly_reasons
                    else []
                ),

            "data_quality_warnings":
                (
                    self.data_quality_warnings.split("||")
                    if self.data_quality_warnings
                    else []
                ),

            # Timestamps
            "created_at":
                (
                    self.created_at.isoformat()
                    if self.created_at
                    else None
                ),

            "updated_at":
                (
                    self.updated_at.isoformat()
                    if self.updated_at
                    else None
                )
        }