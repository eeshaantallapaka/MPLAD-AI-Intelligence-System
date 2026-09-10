import os
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import Config
from database.db import db
from database.models import Project


upload_bp = Blueprint(
    "upload",
    __name__,
    url_prefix="/api"
)


# ============================================================
# COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {
    "project_id": [
        "project_id",
        "projectid",
        "project id",
        "id"
    ],

    "state": [
        "state"
    ],

    "district": [
        "district"
    ],

    "constituency": [
        "constituency",
        "parliamentary_constituency",
        "parliamentary constituency"
    ],

    "mp_name": [
        "mp_name",
        "mp name",
        "mp",
        "member_of_parliament",
        "member of parliament"
    ],

    "project_name": [
        "project_name",
        "project name",
        "project"
    ],

    "project_category": [
        "project_category",
        "project category",
        "category",
        "project_type",
        "project type"
    ],

    "sanctioned_amount": [
        "sanctioned_amount",
        "sanctioned amount",
        "sanction_amount",
        "sanction amount"
    ],

    "estimated_cost": [
        "estimated_cost",
        "estimated cost",
        "estimated_project_cost",
        "estimated project cost"
    ],

    "released_amount": [
        "released_amount",
        "released amount",
        "amount_released",
        "amount released",
        "funds_released",
        "funds released"
    ],

    "expenditure": [
        "expenditure",
        "actual_expenditure",
        "actual expenditure",
        "actual_exp",
        "actual exp"
    ],

    "amount_utilized": [
        "amount_utilized",
        "amount utilized",
        "amount_utilised",
        "amount utilised",
        "utilized_amount",
        "utilised_amount"
    ],

    "project_status": [
        "project_status",
        "project status",
        "status"
    ],

    "sanction_date": [
        "sanction_date",
        "sanction date"
    ],

    "start_date": [
        "start_date",
        "start date",
        "project_start_date",
        "project start date"
    ],

    "expected_completion_date": [
        "expected_completion_date",
        "expected completion date",
        "expected_end_date",
        "expected end date"
    ],

    "completion_date": [
        "completion_date",
        "completion date",
        "actual_completion_date",
        "actual completion date",
        "actual_end_date",
        "actual end date"
    ],

    "implementing_agency": [
        "implementing_agency",
        "implementing agency",
        "agency",
        "implementation_agency",
        "implementation agency"
    ],

    "contractor": [
        "contractor",
        "vendor",
        "vendor_name",
        "vendor name",
        "contractor_name",
        "contractor name"
    ],

    "latitude": [
        "latitude",
        "lat"
    ],

    "longitude": [
        "longitude",
        "long",
        "lng",
        "lon"
    ],

    "beneficiary_count": [
        "beneficiary_count",
        "beneficiary count",
        "beneficiaries",
        "number_of_beneficiaries",
        "number of beneficiaries"
    ]
}


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(column):
    """Convert column names into a consistent format."""

    text = str(column).strip().lower()

    text = text.replace("-", "_")
    text = text.replace("/", "_")
    text = text.replace(" ", "_")

    while "__" in text:
        text = text.replace("__", "_")

    return text


def build_alias_lookup():
    """Build reverse alias lookup."""

    lookup = {}

    for standard_name, aliases in COLUMN_ALIASES.items():

        lookup[
            normalize_column_name(standard_name)
        ] = standard_name

        for alias in aliases:
            lookup[
                normalize_column_name(alias)
            ] = standard_name

    return lookup


def normalize_columns(df):
    """Rename incoming columns to internal standard names."""

    lookup = build_alias_lookup()

    renamed = {}

    for column in df.columns:

        normalized = normalize_column_name(column)

        if normalized in lookup:
            renamed[column] = lookup[normalized]

        else:
            renamed[column] = normalized

    df = df.rename(columns=renamed)

    # Remove duplicate columns created by aliases
    df = df.loc[:, ~df.columns.duplicated()]

    return df


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_text(value):
    """Clean text values."""

    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def clean_numeric(series):
    """Convert a pandas series to numeric values."""

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def clean_date(series):
    """Convert dates safely."""

    return pd.to_datetime(
        series,
        errors="coerce"
    )


# ============================================================
# FILE READING
# ============================================================

def read_uploaded_file(file):
    """Read CSV or Excel upload."""

    filename = secure_filename(
        file.filename or ""
    )

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension == ".csv":

        return pd.read_csv(file)

    if extension in [".xlsx", ".xls"]:

        return pd.read_excel(file)

    raise ValueError(
        "Unsupported file type. "
        "Please upload CSV, XLSX or XLS."
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):
    """Validate uploaded dataset and return warnings."""

    warnings = []

    if df.empty:
        raise ValueError(
            "Uploaded dataset is empty."
        )

    # --------------------------------------------------------
    # Project ID
    # --------------------------------------------------------

    if "project_id" not in df.columns:

        raise ValueError(
            "Missing required column: project_id"
        )

    missing_ids = int(
        df["project_id"]
        .isna()
        .sum()
    )

    if missing_ids > 0:

        warnings.append(
            f"{missing_ids} missing project_id value(s)."
        )

    duplicate_ids = int(
        df["project_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids > 0:

        warnings.append(
            f"{duplicate_ids} duplicate project_id "
            f"value(s) found inside uploaded file."
        )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    numeric_columns = [
        "sanctioned_amount",
        "estimated_cost",
        "released_amount",
        "amount_utilized",
        "expenditure",
        "beneficiary_count",
        "latitude",
        "longitude"
    ]

    for column in numeric_columns:

        if column not in df.columns:
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_count = int(
            (
                df[column].notna()
                & numeric.isna()
            ).sum()
        )

        if invalid_count > 0:

            warnings.append(
                f"{invalid_count} invalid numerical "
                f"value(s) found in '{column}'."
            )

        # Negative financial values
        if column in [
            "sanctioned_amount",
            "estimated_cost",
            "released_amount",
            "amount_utilized",
            "expenditure"
        ]:

            negative_count = int(
                (numeric < 0).sum()
            )

            if negative_count > 0:

                warnings.append(
                    f"{negative_count} negative value(s) "
                    f"found in '{column}'."
                )

    # --------------------------------------------------------
    # Date validation
    # --------------------------------------------------------

    date_columns = [
        "sanction_date",
        "start_date",
        "expected_completion_date",
        "completion_date"
    ]

    for column in date_columns:

        if column not in df.columns:
            continue

        parsed = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        invalid_count = int(
            (
                df[column].notna()
                & parsed.isna()
            ).sum()
        )

        if invalid_count > 0:

            warnings.append(
                f"{invalid_count} invalid date value(s) "
                f"found in '{column}'."
            )

    # --------------------------------------------------------
    # Geographic validation
    # --------------------------------------------------------

    if "latitude" in df.columns:

        lat = pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )

        invalid_lat = int(
            (
                lat.notna()
                & ~lat.between(-90, 90)
            ).sum()
        )

        if invalid_lat > 0:

            warnings.append(
                f"{invalid_lat} invalid latitude "
                f"value(s) found."
            )

    if "longitude" in df.columns:

        lon = pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )

        invalid_lon = int(
            (
                lon.notna()
                & ~lon.between(-180, 180)
            ).sum()
        )

        if invalid_lon > 0:

            warnings.append(
                f"{invalid_lon} invalid longitude "
                f"value(s) found."
            )

    return warnings


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataframe(df):
    """Normalize and prepare dataframe for database insertion."""

    df = df.copy()

    # Normalize columns
    df = normalize_columns(df)

    # --------------------------------------------------------
    # Remove synthetic labels
    # --------------------------------------------------------

    if "_demo_injected_issue" in df.columns:

        df = df.drop(
            columns=["_demo_injected_issue"]
        )

    # --------------------------------------------------------
    # Ensure expected columns exist
    # --------------------------------------------------------

    expected_columns = [
        "project_id",
        "state",
        "district",
        "constituency",
        "mp_name",
        "project_name",
        "project_category",
        "sanctioned_amount",
        "estimated_cost",
        "released_amount",
        "amount_utilized",
        "expenditure",
        "project_status",
        "sanction_date",
        "start_date",
        "expected_completion_date",
        "completion_date",
        "implementing_agency",
        "contractor",
        "latitude",
        "longitude",
        "beneficiary_count"
    ]

    for column in expected_columns:

        if column not in df.columns:
            df[column] = None

    # --------------------------------------------------------
    # Text fields
    # --------------------------------------------------------

    text_columns = [
        "project_id",
        "state",
        "district",
        "constituency",
        "mp_name",
        "project_name",
        "project_category",
        "project_status",
        "implementing_agency",
        "contractor"
    ]

    for column in text_columns:

        df[column] = df[column].apply(
            clean_text
        )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    numeric_columns = [
        "sanctioned_amount",
        "estimated_cost",
        "released_amount",
        "amount_utilized",
        "expenditure",
        "latitude",
        "longitude",
        "beneficiary_count"
    ]

    for column in numeric_columns:

        df[column] = clean_numeric(
            df[column]
        )

    # --------------------------------------------------------
    # Fill sanctioned amount
    # --------------------------------------------------------

    missing_sanction = (
        df["sanctioned_amount"].isna()
        |
        (df["sanctioned_amount"] <= 0)
    )

    estimated_available = (
        df["estimated_cost"].notna()
        &
        (df["estimated_cost"] > 0)
    )

    df.loc[
        missing_sanction & estimated_available,
        "sanctioned_amount"
    ] = df.loc[
        missing_sanction & estimated_available,
        "estimated_cost"
    ]

    # --------------------------------------------------------
    # Fill expenditure
    # --------------------------------------------------------

    missing_expenditure = (
        df["expenditure"].isna()
    )

    utilized_available = (
        df["amount_utilized"].notna()
    )

    df.loc[
        missing_expenditure & utilized_available,
        "expenditure"
    ] = df.loc[
        missing_expenditure & utilized_available,
        "amount_utilized"
    ]

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    date_columns = [
        "sanction_date",
        "start_date",
        "expected_completion_date",
        "completion_date"
    ]

    for column in date_columns:

        df[column] = clean_date(
            df[column]
        )

    return df


# ============================================================
# DATABASE CONVERSION
# ============================================================

def dataframe_row_to_project(row):
    """Convert dataframe row into a Project model."""

    project = Project()

    project.project_id = row.get(
        "project_id"
    )

    project.state = row.get(
        "state"
    )

    project.district = row.get(
        "district"
    )

    project.constituency = row.get(
        "constituency"
    )

    project.mp_name = row.get(
        "mp_name"
    )

    project.project_name = row.get(
        "project_name"
    )

    project.project_category = row.get(
        "project_category"
    )

    project.sanctioned_amount = (
        _safe_float(
            row.get("sanctioned_amount")
        )
    )

    project.released_amount = (
        _safe_float(
            row.get("released_amount")
        )
    )

    project.expenditure = (
        _safe_float(
            row.get("expenditure")
        )
    )

    project.project_status = row.get(
        "project_status"
    )

    project.sanction_date = (
        _safe_datetime(
            row.get("sanction_date")
        )
    )

    project.start_date = (
        _safe_datetime(
            row.get("start_date")
        )
    )

    project.completion_date = (
        _safe_datetime(
            row.get("completion_date")
        )
    )

    project.implementing_agency = row.get(
        "implementing_agency"
    )

    project.contractor = row.get(
        "contractor"
    )

    project.latitude = (
        _safe_float(
            row.get("latitude")
        )
    )

    project.longitude = (
        _safe_float(
            row.get("longitude")
        )
    )

    project.beneficiary_count = (
        _safe_int(
            row.get("beneficiary_count")
        )
    )

    return project


def _safe_float(value):
    """Safely convert to float."""

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):

        return None


def _safe_int(value):
    """Safely convert to integer."""

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

        return int(float(value))

    except (TypeError, ValueError):

        return None


def _safe_datetime(value):
    """Safely convert pandas/numpy datetime."""

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except (TypeError, ValueError):

        pass

    if isinstance(value, pd.Timestamp):

        return value.to_pydatetime()

    if isinstance(value, datetime):

        return value

    try:

        parsed = pd.to_datetime(
            value,
            errors="coerce"
        )

        if pd.isna(parsed):
            return None

        return parsed.to_pydatetime()

    except (TypeError, ValueError):

        return None


# ============================================================
# UPLOAD ENDPOINT
# ============================================================

@upload_bp.route(
    "/upload",
    methods=["POST"]
)
def upload_dataset():

    try:

        # ----------------------------------------------------
        # Check file
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "message": "No file provided."
            }), 400

        file = request.files["file"]

        if not file or not file.filename:

            return jsonify({
                "success": False,
                "message": "No file selected."
            }), 400

        filename = secure_filename(
            file.filename
        )

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in Config.ALLOWED_EXTENSIONS:

            return jsonify({
                "success": False,
                "message": (
                    "Unsupported file type. "
                    "Allowed formats: CSV, XLSX, XLS."
                )
            }), 400

        # ----------------------------------------------------
        # Read file
        # ----------------------------------------------------

        df = read_uploaded_file(file)

        rows_received = len(df)

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        warnings = validate_dataset(
            df
        )

        # ----------------------------------------------------
        # Prepare
        # ----------------------------------------------------

        df = prepare_dataframe(
            df
        )

        # ----------------------------------------------------
        # Remove rows with missing IDs
        # ----------------------------------------------------

        df = df[
            df["project_id"].notna()
        ].copy()

        # ----------------------------------------------------
        # Track results
        # ----------------------------------------------------

        records_inserted = 0
        records_updated = 0
        records_skipped = 0

        # ----------------------------------------------------
        # Process every row
        # ----------------------------------------------------

        for _, row in df.iterrows():

            project_id = str(
                row.get("project_id")
            ).strip()

            if not project_id:
                records_skipped += 1
                continue

            # ------------------------------------------------
            # Find existing project
            # ------------------------------------------------

            existing_project = (
                Project.query
                .filter_by(
                    project_id=project_id
                )
                .first()
            )

            # =================================================
            # EXISTING PROJECT → UPDATE IT
            # =================================================

            if existing_project is not None:

                new_project = (
                    dataframe_row_to_project(
                        row
                    )
                )

                # Preserve database identity
                existing_project.state = (
                    new_project.state
                )

                existing_project.district = (
                    new_project.district
                )

                existing_project.constituency = (
                    new_project.constituency
                )

                existing_project.mp_name = (
                    new_project.mp_name
                )

                existing_project.project_name = (
                    new_project.project_name
                )

                existing_project.project_category = (
                    new_project.project_category
                )

                existing_project.sanctioned_amount = (
                    new_project.sanctioned_amount
                )

                existing_project.released_amount = (
                    new_project.released_amount
                )

                existing_project.expenditure = (
                    new_project.expenditure
                )

                existing_project.project_status = (
                    new_project.project_status
                )

                existing_project.sanction_date = (
                    new_project.sanction_date
                )

                existing_project.start_date = (
                    new_project.start_date
                )

                existing_project.completion_date = (
                    new_project.completion_date
                )

                existing_project.implementing_agency = (
                    new_project.implementing_agency
                )

                existing_project.contractor = (
                    new_project.contractor
                )

                existing_project.latitude = (
                    new_project.latitude
                )

                existing_project.longitude = (
                    new_project.longitude
                )

                existing_project.beneficiary_count = (
                    new_project.beneficiary_count
                )

                # Reset previous analysis results
                existing_project.anomaly_score = 0
                existing_project.risk_score = 0
                existing_project.risk_level = "NORMAL"
                existing_project.financial_indicator = 0
                existing_project.delay_indicator = 0
                existing_project.duplicate_indicator = 0
                existing_project.contractor_indicator = 0
                existing_project.geographic_indicator = 0
                existing_project.data_quality_indicator = 0
                existing_project.anomaly_reasons = None
                existing_project.data_quality_warnings = None

                records_updated += 1

            # =================================================
            # NEW PROJECT → INSERT IT
            # =================================================

            else:

                project = dataframe_row_to_project(
                    row
                )

                db.session.add(
                    project
                )

                records_inserted += 1

        # ----------------------------------------------------
        # Save transaction
        # ----------------------------------------------------

        db.session.commit()

        # ----------------------------------------------------
        # Add date-cleaning warnings
        # ----------------------------------------------------

        for column in [
            "sanction_date",
            "start_date",
            "expected_completion_date",
            "completion_date"
        ]:

            if column not in df.columns:
                continue

            original = df[column]

            invalid_dates = int(
                original.isna().sum()
            )

            if invalid_dates > 0:

                # Only report if original dataset
                # contained values for this field.
                warnings.append(
                    f"{invalid_dates} invalid/missing "
                    f"date value(s) retained as missing "
                    f"in '{column}'."
                )

        # ----------------------------------------------------
        # Remove duplicate warning messages
        # ----------------------------------------------------

        warnings = list(
            dict.fromkeys(warnings)
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "message":
                "Dataset uploaded and processed successfully.",

            "filename":
                filename,

            "rows_received":
                int(rows_received),

            "records_inserted":
                int(records_inserted),

            "records_updated":
                int(records_updated),

            "records_skipped":
                int(records_skipped),

            "detected_columns":
                sorted(
                    list(df.columns)
                ),

            "warnings":
                warnings
        })

    except Exception as error:

        db.session.rollback()

        return jsonify({

            "success": False,

            "message":
                "Dataset upload failed.",

            "error":
                str(error)

        }), 500