import re

import numpy as np
import pandas as pd

from database.db import db
from database.models import Project

from ml.anomaly_detector import detect_anomalies
from analysis.financial_anomaly import analyze_financial_anomalies
from analysis.delay_detection import analyze_delay_anomalies


# ============================================================
# LOAD PROJECTS
# ============================================================

def load_projects_dataframe():
    projects = Project.query.all()

    if not projects:
        return pd.DataFrame()

    records = []

    for project in projects:
        records.append({
            "project_id": project.project_id,
            "state": project.state,
            "district": project.district,
            "constituency": project.constituency,
            "mp_name": project.mp_name,
            "project_name": project.project_name,
            "project_category": project.project_category,
            "sanctioned_amount": project.sanctioned_amount,
            "released_amount": project.released_amount,
            "expenditure": project.expenditure,
            "project_status": project.project_status,
            "sanction_date": project.sanction_date,
            "start_date": project.start_date,
            "completion_date": project.completion_date,
            "implementing_agency": project.implementing_agency,
            "contractor": project.contractor,
            "latitude": project.latitude,
            "longitude": project.longitude,
            "beneficiary_count": project.beneficiary_count,
        })

    return pd.DataFrame(records)


# ============================================================
# HELPERS
# ============================================================

def _safe_number(value, default=0.0):
    try:
        if value is None:
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def _risk_level(score):
    score = _safe_number(score)

    if score >= 75:
        return "CRITICAL"

    if score >= 50:
        return "HIGH"

    if score >= 25:
        return "MEDIUM"

    if score > 0:
        return "LOW"

    return "NORMAL"


def _combine_reasons(*reason_groups):
    reasons = []

    for group in reason_groups:

        if group is None:
            continue

        try:
            if pd.isna(group):
                continue
        except (TypeError, ValueError):
            pass

        text = str(group).strip()

        if not text:
            continue

        for reason in text.split("||"):

            reason = reason.strip()

            if reason and reason not in reasons:
                reasons.append(reason)

    return " || ".join(reasons)


def _extract_detection_dataframe(detection_result):

    if isinstance(detection_result, pd.DataFrame):
        return detection_result

    if isinstance(detection_result, dict):

        if isinstance(
            detection_result.get("data"),
            pd.DataFrame
        ):
            return detection_result["data"]

        if isinstance(
            detection_result.get("df"),
            pd.DataFrame
        ):
            return detection_result["df"]

    if isinstance(detection_result, tuple):

        for item in detection_result:

            if isinstance(item, pd.DataFrame):
                return item

    raise TypeError(
        "detect_anomalies() did not return a DataFrame."
    )


def _normalize_text(value):

    if value is None:
        return ""

    text = str(value).lower().strip()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def analyze_duplicate_anomalies(feature_df):

    result = feature_df.copy()

    result["duplicate_indicator"] = 0.0
    result["duplicate_reasons"] = ""

    if len(result) <= 1:
        return result

    required = [
        "project_name",
        "district",
        "project_category",
        "sanctioned_amount"
    ]

    for column in required:

        if column not in result.columns:
            result[column] = ""

    result["_dup_name"] = (
        result["project_name"]
        .fillna("")
        .astype(str)
        .map(_normalize_text)
    )

    result["_dup_district"] = (
        result["district"]
        .fillna("")
        .astype(str)
        .map(_normalize_text)
    )

    result["_dup_category"] = (
        result["project_category"]
        .fillna("")
        .astype(str)
        .map(_normalize_text)
    )

    result["_dup_amount"] = pd.to_numeric(
        result["sanctioned_amount"],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # Exact duplicate project names
    # --------------------------------------------------------

    valid_names = result[
        result["_dup_name"] != ""
    ]

    name_counts = (
        valid_names
        .groupby("_dup_name")
        .size()
    )

    duplicate_names = name_counts[
        name_counts > 1
    ]

    for name, count in duplicate_names.items():

        indexes = list(
            result.index[
                result["_dup_name"] == name
            ]
        )

        for index in indexes:

            result.loc[
                index,
                "duplicate_indicator"
            ] = 100.0

            result.loc[
                index,
                "duplicate_reasons"
            ] = (
                "Exact project-name duplicate pattern "
                f"detected ({count} matching records)."
            )

    # --------------------------------------------------------
    # Grouped similarity
    # --------------------------------------------------------

    result["_dup_group"] = (
        result["_dup_district"]
        + "|"
        + result["_dup_category"]
    )

    grouped = result.groupby(
        "_dup_group",
        sort=False
    )

    for _, group in grouped:

        if len(group) < 2:
            continue

        indexes = list(group.index)

        # Prevent pathological large groups.
        if len(indexes) > 100:
            indexes = indexes[:100]

        names = {
            index: result.loc[
                index,
                "_dup_name"
            ]
            for index in indexes
        }

        amounts = {
            index: _safe_number(
                result.loc[
                    index,
                    "_dup_amount"
                ]
            )
            for index in indexes
        }

        for i in range(len(indexes)):

            index_a = indexes[i]
            name_a = names[index_a]

            if not name_a:
                continue

            tokens_a = set(
                name_a.split()
            )

            if not tokens_a:
                continue

            best_score = 0.0

            for j in range(
                i + 1,
                len(indexes)
            ):

                index_b = indexes[j]
                name_b = names[index_b]

                if not name_b:
                    continue

                tokens_b = set(
                    name_b.split()
                )

                if not tokens_b:
                    continue

                union = (
                    tokens_a | tokens_b
                )

                if not union:
                    continue

                intersection = (
                    tokens_a & tokens_b
                )

                token_similarity = (
                    len(intersection)
                    /
                    len(union)
                )

                amount_bonus = 0.0

                amount_a = amounts[index_a]
                amount_b = amounts[index_b]

                if (
                    amount_a > 0
                    and amount_b > 0
                ):

                    average = (
                        amount_a
                        + amount_b
                    ) / 2

                    if average > 0:

                        difference = abs(
                            amount_a
                            - amount_b
                        )

                        ratio = (
                            difference
                            /
                            average
                        )

                        if ratio <= 0.10:
                            amount_bonus = 0.15

                combined = min(
                    1.0,
                    token_similarity
                    + amount_bonus
                )

                if combined > best_score:
                    best_score = combined

            if best_score >= 0.75:

                score = min(
                    100,
                    max(
                        0,
                        (best_score - 0.65) * 285
                    )
                )

                current_score = _safe_number(
                    result.loc[
                        index_a,
                        "duplicate_indicator"
                    ]
                )

                if score > current_score:

                    result.loc[
                        index_a,
                        "duplicate_indicator"
                    ] = round(
                        score,
                        2
                    )

                    result.loc[
                        index_a,
                        "duplicate_reasons"
                    ] = (
                        "Highly similar project pattern "
                        "detected within the same "
                        "district/category."
                    )

    result.drop(
        columns=[
            "_dup_name",
            "_dup_district",
            "_dup_category",
            "_dup_amount",
            "_dup_group"
        ],
        inplace=True,
        errors="ignore"
    )

    return result


# ============================================================
# CONTRACTOR ANALYSIS
# ============================================================

def analyze_contractor_anomalies(feature_df):

    result = feature_df.copy()

    result["contractor_indicator"] = 0.0
    result["contractor_reasons"] = ""

    if "contractor" not in result.columns:
        return result

    contractors = (
        result["contractor"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid = contractors[
        contractors != ""
    ]

    if valid.empty:
        return result

    counts = valid.value_counts()
    total = len(valid)

    concentrations = (
        contractors
        .map(counts)
        .fillna(0)
        / total
    )

    # IMPORTANT:
    # Assign the complete Series, not a filtered subset.

    result["contractor_indicator"] = np.select(
        [
            concentrations >= 0.50,
            concentrations >= 0.30,
            concentrations >= 0.20,
            concentrations >= 0.10
        ],
        [
            100,
            75,
            50,
            20
        ],
        default=0
    )

    for index in result.index:

        contractor = contractors.loc[index]

        if not contractor:
            continue

        concentration = _safe_number(
            concentrations.loc[index]
        )

        count = int(
            counts.get(
                contractor,
                0
            )
        )

        if concentration >= 0.50:

            reason = (
                "Contractor concentration is unusually high "
                f"({count} of {total} projects)."
            )

        elif concentration >= 0.30:

            reason = (
                "High contractor concentration detected "
                f"({count} of {total} projects)."
            )

        elif concentration >= 0.20:

            reason = (
                "Elevated contractor concentration detected "
                f"({count} of {total} projects)."
            )

        elif concentration >= 0.10:

            reason = (
                "Contractor appears across multiple projects."
            )

        else:
            reason = ""

        result.loc[
            index,
            "contractor_reasons"
        ] = reason

    return result


# ============================================================
# GEOGRAPHIC ANALYSIS
# ============================================================

def analyze_geographic_anomalies(feature_df):

    result = feature_df.copy()

    result["geographic_indicator"] = 0.0
    result["geographic_reasons"] = ""

    if (
        "latitude" not in result.columns
        or "longitude" not in result.columns
    ):
        return result

    lat = pd.to_numeric(
        result["latitude"],
        errors="coerce"
    )

    lon = pd.to_numeric(
        result["longitude"],
        errors="coerce"
    )

    valid = (
        lat.notna()
        &
        lon.notna()
    )

    if valid.sum() < 3:
        return result

    # --------------------------------------------------------
    # Geographic grid
    # --------------------------------------------------------

    grid_lat = (
        lat
        .div(0.05)
        .round()
    )

    grid_lon = (
        lon
        .div(0.05)
        .round()
    )

    grid_key = (
        grid_lat
        .fillna(-999999)
        .astype(int)
        .astype(str)
        + "_"
        +
        grid_lon
        .fillna(-999999)
        .astype(int)
        .astype(str)
    )

    counts = (
        grid_key[valid]
        .value_counts()
    )

    nearby_counts = (
        grid_key
        .map(counts)
        .fillna(0)
        - 1
    )

    nearby_counts = nearby_counts.clip(
        lower=0
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Assign complete Series.
    # --------------------------------------------------------

    result["geographic_indicator"] = np.select(
        [
            nearby_counts >= 10,
            nearby_counts >= 6,
            nearby_counts >= 3
        ],
        [
            100,
            75,
            45
        ],
        default=0
    )

    for index in result.index:

        if not valid.loc[index]:
            continue

        nearby = int(
            nearby_counts.loc[index]
        )

        if nearby >= 10:

            reason = (
                "Very dense geographic project cluster "
                f"detected ({nearby} nearby projects)."
            )

        elif nearby >= 6:

            reason = (
                "High geographic project concentration "
                f"detected ({nearby} nearby projects)."
            )

        elif nearby >= 3:

            reason = (
                "Multiple nearby projects detected "
                f"({nearby} nearby projects)."
            )

        else:

            reason = ""

        result.loc[
            index,
            "geographic_reasons"
        ] = reason

    return result


# ============================================================
# DATA QUALITY
# ============================================================

def analyze_data_quality(feature_df):

    result = feature_df.copy()

    result["data_quality_indicator"] = 0.0
    result["data_quality_reasons"] = ""

    required_fields = [
        (
            "state",
            "State is missing."
        ),
        (
            "district",
            "District is missing."
        ),
        (
            "project_name",
            "Project name is missing."
        ),
        (
            "project_status",
            "Project status is missing."
        )
    ]

    for index, row in result.iterrows():

        score = 0
        reasons = []

        # ----------------------------------------------------
        # Required text
        # ----------------------------------------------------

        for field, reason in required_fields:

            value = row.get(field)

            if (
                value is None
                or str(value).strip() == ""
                or str(value).lower() == "nan"
            ):

                score += 15
                reasons.append(reason)

        # ----------------------------------------------------
        # Financial values
        # ----------------------------------------------------

        sanctioned = _safe_number(
            row.get("sanctioned_amount")
        )

        released = _safe_number(
            row.get("released_amount")
        )

        expenditure = _safe_number(
            row.get("expenditure")
        )

        if sanctioned < 0:

            score += 30

            reasons.append(
                "Negative sanctioned amount."
            )

        if released < 0:

            score += 20

            reasons.append(
                "Negative released amount."
            )

        if expenditure < 0:

            score += 25

            reasons.append(
                "Negative expenditure."
            )

        # ----------------------------------------------------
        # Beneficiaries
        # ----------------------------------------------------

        beneficiaries = row.get(
            "beneficiary_count"
        )

        if beneficiaries is not None:

            beneficiaries = _safe_number(
                beneficiaries
            )

            if beneficiaries < 0:

                score += 30

                reasons.append(
                    "Negative beneficiary count."
                )

        # ----------------------------------------------------
        # Geographic validation
        # ----------------------------------------------------

        latitude = row.get("latitude")
        longitude = row.get("longitude")

        latitude = _safe_number(
            latitude,
            None
        )

        longitude = _safe_number(
            longitude,
            None
        )

        if latitude is not None:

            if (
                latitude < -90
                or latitude > 90
            ):

                score += 25

                reasons.append(
                    "Invalid latitude value."
                )

        if longitude is not None:

            if (
                longitude < -180
                or longitude > 180
            ):

                score += 25

                reasons.append(
                    "Invalid longitude value."
                )

        result.loc[
            index,
            "data_quality_indicator"
        ] = min(
            score,
            100
        )

        result.loc[
            index,
            "data_quality_reasons"
        ] = " || ".join(reasons)

    return result


# ============================================================
# COMBINED RISK
# ============================================================

def calculate_combined_risk(feature_df):

    result = feature_df.copy()

    indicators = [
        "anomaly_score",
        "financial_indicator",
        "delay_indicator",
        "duplicate_indicator",
        "contractor_indicator",
        "geographic_indicator",
        "data_quality_indicator"
    ]

    for column in indicators:

        if column not in result.columns:
            result[column] = 0.0

        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 100)
        )

    # --------------------------------------------------------
    # Transparent weighted risk score
    # --------------------------------------------------------

    result["risk_score"] = (
        result["anomaly_score"] * 0.30
        +
        result["financial_indicator"] * 0.20
        +
        result["delay_indicator"] * 0.15
        +
        result["duplicate_indicator"] * 0.15
        +
        result["contractor_indicator"] * 0.10
        +
        result["geographic_indicator"] * 0.05
        +
        result["data_quality_indicator"] * 0.05
    )

    result["risk_score"] = (
        result["risk_score"]
        .clip(0, 100)
        .round(2)
    )

    result["risk_level"] = (
        result["risk_score"]
        .apply(_risk_level)
    )

    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_anomaly_results(feature_df):

    if (
        feature_df is None
        or feature_df.empty
    ):
        return 0

    # Load all projects once.
    projects = Project.query.all()

    project_map = {
        str(project.project_id).strip():
            project
        for project in projects
    }

    updated = 0

    for _, row in feature_df.iterrows():

        project_id = str(
            row.get(
                "project_id",
                ""
            )
        ).strip()

        if not project_id:
            continue

        project = project_map.get(
            project_id
        )

        if project is None:
            continue

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        project.anomaly_score = round(
            _safe_number(
                row.get(
                    "anomaly_score"
                )
            ),
            2
        )

        project.financial_indicator = round(
            _safe_number(
                row.get(
                    "financial_indicator"
                )
            ),
            2
        )

        project.delay_indicator = round(
            _safe_number(
                row.get(
                    "delay_indicator"
                )
            ),
            2
        )

        project.duplicate_indicator = round(
            _safe_number(
                row.get(
                    "duplicate_indicator"
                )
            ),
            2
        )

        project.contractor_indicator = round(
            _safe_number(
                row.get(
                    "contractor_indicator"
                )
            ),
            2
        )

        project.geographic_indicator = round(
            _safe_number(
                row.get(
                    "geographic_indicator"
                )
            ),
            2
        )

        project.data_quality_indicator = round(
            _safe_number(
                row.get(
                    "data_quality_indicator"
                )
            ),
            2
        )

        project.risk_score = round(
            _safe_number(
                row.get(
                    "risk_score"
                )
            ),
            2
        )

        project.risk_level = _risk_level(
            project.risk_score
        )

        # ----------------------------------------------------
        # AI reason
        # ----------------------------------------------------

        ai_reason = ""

        is_anomaly = int(
            _safe_number(
                row.get(
                    "is_anomaly"
                )
            )
        )

        if is_anomaly == 1:

            ai_reason = (
                "Isolation Forest detected an unusual "
                "multivariate project pattern."
            )

        # ----------------------------------------------------
        # Combined explanations
        # ----------------------------------------------------

        project.anomaly_reasons = (
            _combine_reasons(

                ai_reason,

                row.get(
                    "financial_reasons",
                    ""
                ),

                row.get(
                    "delay_reasons",
                    ""
                ),

                row.get(
                    "duplicate_reasons",
                    ""
                ),

                row.get(
                    "contractor_reasons",
                    ""
                ),

                row.get(
                    "geographic_reasons",
                    ""
                ),

                row.get(
                    "data_quality_reasons",
                    ""
                )
            )
        )

        if not project.anomaly_reasons:
            project.anomaly_reasons = None

        updated += 1

    db.session.commit()

    return updated


# ============================================================
# COMPLETE AI PIPELINE
# ============================================================

def run_anomaly_pipeline():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_projects_dataframe()

    if df.empty:

        return {
            "success": False,
            "message":
                "No projects found in database.",
            "projects_analyzed": 0,
            "anomalies_detected": 0,
            "financial_anomalies": 0,
            "delayed_projects": 0
        }

    # --------------------------------------------------------
    # STEP 1 - MACHINE LEARNING
    # --------------------------------------------------------

    detection_result = detect_anomalies(
        df,
        save_model=True
    )

    feature_df = (
        _extract_detection_dataframe(
            detection_result
        )
    )

    # --------------------------------------------------------
    # STEP 2 - FINANCIAL
    # --------------------------------------------------------

    feature_df = (
        analyze_financial_anomalies(
            feature_df
        )
    )

    # --------------------------------------------------------
    # STEP 3 - DELAY
    # --------------------------------------------------------

    feature_df = (
        analyze_delay_anomalies(
            feature_df
        )
    )

    # --------------------------------------------------------
    # STEP 4 - DUPLICATE
    # --------------------------------------------------------

    feature_df = (
        analyze_duplicate_anomalies(
            feature_df
        )
    )

    # --------------------------------------------------------
    # STEP 5 - CONTRACTOR
    # --------------------------------------------------------

    feature_df = (
        analyze_contractor_anomalies(
            feature_df
        )
    )

    # --------------------------------------------------------
    # STEP 6 - GEOGRAPHIC
    # --------------------------------------------------------

    feature_df = (
        analyze_geographic_anomalies(
            feature_df
        )
    )

    # --------------------------------------------------------
    # STEP 7 - DATA QUALITY
    # --------------------------------------------------------

    feature_df = (
        analyze_data_quality(
            feature_df
        )
    )

    # --------------------------------------------------------
    # STEP 8 - COMBINED RISK
    # --------------------------------------------------------

    feature_df = (
        calculate_combined_risk(
            feature_df
        )
    )

    # --------------------------------------------------------
    # STEP 9 - SAVE
    # --------------------------------------------------------

    updated = save_anomaly_results(
        feature_df
    )

    # --------------------------------------------------------
    # STEP 10 - SUMMARY
    # --------------------------------------------------------

    if "is_anomaly" in feature_df.columns:

        anomalies_detected = int(
            pd.to_numeric(
                feature_df["is_anomaly"],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    else:

        anomalies_detected = 0

    financial_anomalies = int(
        (
            feature_df[
                "financial_indicator"
            ] > 0
        ).sum()
    )

    delayed_projects = int(
        (
            feature_df[
                "delay_indicator"
            ] > 0
        ).sum()
    )

    duplicate_anomalies = int(
        (
            feature_df[
                "duplicate_indicator"
            ] > 0
        ).sum()
    )

    contractor_anomalies = int(
        (
            feature_df[
                "contractor_indicator"
            ] > 0
        ).sum()
    )

    geographic_anomalies = int(
        (
            feature_df[
                "geographic_indicator"
            ] > 0
        ).sum()
    )

    data_quality_anomalies = int(
        (
            feature_df[
                "data_quality_indicator"
            ] > 0
        ).sum()
    )

    high_risk = int(
        (
            feature_df[
                "risk_score"
            ] >= 50
        ).sum()
    )

    critical_risk = int(
        (
            feature_df[
                "risk_score"
            ] >= 75
        ).sum()
    )

    return {

        "success": True,

        "message":
            "AI anomaly detection completed successfully.",

        "projects_analyzed":
            int(len(feature_df)),

        "records_updated":
            int(updated),

        "anomalies_detected":
            anomalies_detected,

        "financial_anomalies":
            financial_anomalies,

        "delayed_projects":
            delayed_projects,

        "duplicate_anomalies":
            duplicate_anomalies,

        "contractor_anomalies":
            contractor_anomalies,

        "geographic_anomalies":
            geographic_anomalies,

        "data_quality_anomalies":
            data_quality_anomalies,

        "high_risk_projects":
            high_risk,

        "critical_risk_projects":
            critical_risk,

        "average_anomaly_score":
            round(
                float(
                    feature_df[
                        "anomaly_score"
                    ].mean()
                ),
                2
            ),

        "average_financial_indicator":
            round(
                float(
                    feature_df[
                        "financial_indicator"
                    ].mean()
                ),
                2
            ),

        "average_delay_indicator":
            round(
                float(
                    feature_df[
                        "delay_indicator"
                    ].mean()
                ),
                2
            ),

        "average_risk_score":
            round(
                float(
                    feature_df[
                        "risk_score"
                    ].mean()
                ),
                2
            )
    }


# ============================================================
# STORED SUMMARY
# ============================================================

def get_anomaly_summary():

    total = Project.query.count()

    anomalies = Project.query.filter(
        Project.anomaly_score > 0
    ).count()

    financial = Project.query.filter(
        Project.financial_indicator > 0
    ).count()

    delayed = Project.query.filter(
        Project.delay_indicator > 0
    ).count()

    duplicates = Project.query.filter(
        Project.duplicate_indicator > 0
    ).count()

    contractors = Project.query.filter(
        Project.contractor_indicator > 0
    ).count()

    geographic = Project.query.filter(
        Project.geographic_indicator > 0
    ).count()

    data_quality = Project.query.filter(
        Project.data_quality_indicator > 0
    ).count()

    high_risk = Project.query.filter(
        Project.risk_score >= 50
    ).count()

    critical = Project.query.filter(
        Project.risk_score >= 75
    ).count()

    average_risk = db.session.query(
        db.func.avg(
            Project.risk_score
        )
    ).scalar()

    average_anomaly = db.session.query(
        db.func.avg(
            Project.anomaly_score
        )
    ).scalar()

    return {

        "success": True,

        "total_projects":
            int(total),

        "projects_with_anomalies":
            int(anomalies),

        "financial_anomalies":
            int(financial),

        "delayed_projects":
            int(delayed),

        "duplicate_anomalies":
            int(duplicates),

        "contractor_anomalies":
            int(contractors),

        "geographic_anomalies":
            int(geographic),

        "data_quality_anomalies":
            int(data_quality),

        "high_risk_projects":
            int(high_risk),

        "critical_risk_projects":
            int(critical),

        "average_risk_score":
            round(
                float(
                    average_risk or 0
                ),
                2
            ),

        "average_anomaly_score":
            round(
                float(
                    average_anomaly or 0
                ),
                2
            )
    }