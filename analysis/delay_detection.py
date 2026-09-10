import pandas as pd
import numpy as np

from config import Config


def _safe_datetime(series):
    """Safely convert a pandas Series to datetime."""

    return pd.to_datetime(
        series,
        errors="coerce"
    )


def _safe_numeric(series):
    """Safely convert a pandas Series to numeric."""

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def _combine_reason(existing, new_reason):
    """Add a reason without creating duplicate text."""

    if not new_reason:
        return existing

    if not existing or str(existing).lower() == "nan":
        return new_reason

    existing_parts = [
        part.strip()
        for part in str(existing).split("||")
        if part.strip()
    ]

    if new_reason not in existing_parts:
        existing_parts.append(new_reason)

    return " || ".join(existing_parts)


def _delay_severity(score):
    """Convert delay indicator into a severity level."""

    if score >= 75:
        return "CRITICAL"

    if score >= 50:
        return "HIGH"

    if score >= 25:
        return "MEDIUM"

    if score > 0:
        return "LOW"

    return "NORMAL"


def analyze_delay_anomalies(df):
    """
    Detect explainable project-delay anomalies.

    This function identifies projects that may require
    administrative review.

    It does NOT declare fraud.
    """

    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    result = df.copy()

    # ========================================================
    # Ensure required columns exist
    # ========================================================

    required_columns = [
        "start_date",
        "completion_date",
        "expected_completion_date",
        "project_status"
    ]

    for column in required_columns:

        if column not in result.columns:
            result[column] = None

    # ========================================================
    # Convert dates
    # ========================================================

    result["start_date"] = _safe_datetime(
        result["start_date"]
    )

    result["completion_date"] = _safe_datetime(
        result["completion_date"]
    )

    result["expected_completion_date"] = _safe_datetime(
        result["expected_completion_date"]
    )

    # ========================================================
    # Current date
    # ========================================================

    today = pd.Timestamp.today().normalize()

    # ========================================================
    # Determine project status
    # ========================================================

    result["status_normalized"] = (
        result["project_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    completed_statuses = {
        "completed",
        "complete",
        "closed",
        "finished"
    }

    ongoing_statuses = {
        "ongoing",
        "in progress",
        "in_progress",
        "under construction",
        "active",
        "started"
    }

    result["is_completed_project"] = (
        result["status_normalized"]
        .isin(completed_statuses)
    )

    result["is_ongoing_project"] = (
        result["status_normalized"]
        .isin(ongoing_statuses)
    )

    # ========================================================
    # Expected duration
    # ========================================================

    result["expected_duration_days"] = (
        result["expected_completion_date"]
        - result["start_date"]
    ).dt.days

    default_duration = getattr(
        Config,
        "DEFAULT_EXPECTED_DURATION_DAYS",
        180
    )

    result["expected_duration_days"] = (
        result["expected_duration_days"]
        .fillna(default_duration)
    )

    result["expected_duration_days"] = (
        result["expected_duration_days"]
        .clip(lower=1)
    )

    # ========================================================
    # Actual duration
    # ========================================================

    result["actual_duration_days"] = np.nan

    completed_mask = (
        result["start_date"].notna()
        &
        result["completion_date"].notna()
    )

    result.loc[
        completed_mask,
        "actual_duration_days"
    ] = (
        result.loc[
            completed_mask,
            "completion_date"
        ]
        -
        result.loc[
            completed_mask,
            "start_date"
        ]
    ).dt.days

    # For ongoing projects, calculate elapsed duration
    ongoing_mask = (
        result["start_date"].notna()
        &
        result["is_ongoing_project"]
    )

    result.loc[
        ongoing_mask,
        "actual_duration_days"
    ] = (
        today
        -
        result.loc[
            ongoing_mask,
            "start_date"
        ]
    ).dt.days

    # ========================================================
    # Delay based on expected completion date
    # ========================================================

    result["delay_days"] = 0.0

    # Completed projects:
    # actual completion later than expected
    completed_delay_mask = (
        result["is_completed_project"]
        &
        result["completion_date"].notna()
        &
        result["expected_completion_date"].notna()
        &
        (
            result["completion_date"]
            >
            result["expected_completion_date"]
        )
    )

    result.loc[
        completed_delay_mask,
        "delay_days"
    ] = (
        result.loc[
            completed_delay_mask,
            "completion_date"
        ]
        -
        result.loc[
            completed_delay_mask,
            "expected_completion_date"
        ]
    ).dt.days

    # Ongoing projects:
    # today later than expected completion
    ongoing_delay_mask = (
        result["is_ongoing_project"]
        &
        result["expected_completion_date"].notna()
        &
        (
            today
            >
            result["expected_completion_date"]
        )
    )

    result.loc[
        ongoing_delay_mask,
        "delay_days"
    ] = (
        today
        -
        result.loc[
            ongoing_delay_mask,
            "expected_completion_date"
        ]
    ).dt.days

    # ========================================================
    # Duration overrun
    # ========================================================

    result["duration_overrun_days"] = (
        result["actual_duration_days"]
        -
        result["expected_duration_days"]
    )

    result["duration_overrun_days"] = (
        result["duration_overrun_days"]
        .fillna(0)
        .clip(lower=0)
    )

    # ========================================================
    # Delay ratio
    # ========================================================

    result["delay_ratio"] = (
        result["delay_days"]
        /
        result["expected_duration_days"]
    )

    result["delay_ratio"] = (
        result["delay_ratio"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .clip(lower=0)
    )

    # ========================================================
    # Initialize indicator
    # ========================================================

    result["delay_indicator"] = 0.0

    result["delay_reasons"] = ""

    # ========================================================
    # Rule 1 — Any confirmed delay
    # ========================================================

    any_delay = (
        result["delay_days"] > 0
    )

    for index in result.index[any_delay]:

        days = int(
            result.at[index, "delay_days"]
        )

        result.at[index, "delay_indicator"] += 20

        result.at[index, "delay_reasons"] = (
            _combine_reason(
                result.at[index, "delay_reasons"],
                f"Project delayed by {days} day(s)."
            )
        )

    # ========================================================
    # Rule 2 — Delay > 30 days
    # ========================================================

    delay_30 = (
        result["delay_days"] > 30
    )

    for index in result.index[delay_30]:

        result.at[index, "delay_indicator"] += 15

        result.at[index, "delay_reasons"] = (
            _combine_reason(
                result.at[index, "delay_reasons"],
                "Project delay exceeds 30 days."
            )
        )

    # ========================================================
    # Rule 3 — Delay > 90 days
    # ========================================================

    delay_90 = (
        result["delay_days"] > 90
    )

    for index in result.index[delay_90]:

        result.at[index, "delay_indicator"] += 20

        result.at[index, "delay_reasons"] = (
            _combine_reason(
                result.at[index, "delay_reasons"],
                "Project delay exceeds 90 days."
            )
        )

    # ========================================================
    # Rule 4 — Delay > 180 days
    # ========================================================

    delay_180 = (
        result["delay_days"] > 180
    )

    for index in result.index[delay_180]:

        result.at[index, "delay_indicator"] += 25

        result.at[index, "delay_reasons"] = (
            _combine_reason(
                result.at[index, "delay_reasons"],
                "Project delay exceeds 180 days."
            )
        )

    # ========================================================
    # Rule 5 — Duration overrun > 25%
    # ========================================================

    duration_overrun = (
        result["actual_duration_days"].notna()
        &
        (
            result["actual_duration_days"]
            >
            result["expected_duration_days"] * 1.25
        )
    )

    for index in result.index[duration_overrun]:

        result.at[index, "delay_indicator"] += 15

        result.at[index, "delay_reasons"] = (
            _combine_reason(
                result.at[index, "delay_reasons"],
                "Actual project duration exceeds "
                "expected duration by more than 25%."
            )
        )

    # ========================================================
    # Rule 6 — Duration overrun > 50%
    # ========================================================

    severe_duration_overrun = (
        result["actual_duration_days"].notna()
        &
        (
            result["actual_duration_days"]
            >
            result["expected_duration_days"] * 1.50
        )
    )

    for index in result.index[severe_duration_overrun]:

        result.at[index, "delay_indicator"] += 20

        result.at[index, "delay_reasons"] = (
            _combine_reason(
                result.at[index, "delay_reasons"],
                "Actual project duration exceeds "
                "expected duration by more than 50%."
            )
        )

    # ========================================================
    # Rule 7 — Ongoing project past expected completion
    # ========================================================

    overdue_ongoing = (
        result["is_ongoing_project"]
        &
        result["expected_completion_date"].notna()
        &
        (
            today
            >
            result["expected_completion_date"]
        )
    )

    for index in result.index[overdue_ongoing]:

        result.at[index, "delay_indicator"] += 20

        result.at[index, "delay_reasons"] = (
            _combine_reason(
                result.at[index, "delay_reasons"],
                "Ongoing project has passed its "
                "expected completion date."
            )
        )

    # ========================================================
    # Rule 8 — Very high delay ratio
    # ========================================================

    extreme_delay_ratio = (
        result["delay_ratio"] >= 1.0
    )

    for index in result.index[extreme_delay_ratio]:

        result.at[index, "delay_indicator"] += 20

        result.at[index, "delay_reasons"] = (
            _combine_reason(
                result.at[index, "delay_reasons"],
                "Delay is at least equal to the "
                "planned project duration."
            )
        )

    # ========================================================
    # Cap indicator
    # ========================================================

    result["delay_indicator"] = (
        result["delay_indicator"]
        .clip(
            lower=0,
            upper=100
        )
    )

    # ========================================================
    # Severity
    # ========================================================

    result["delay_severity"] = (
        result["delay_indicator"]
        .apply(_delay_severity)
    )

    # ========================================================
    # Clean helper fields
    # ========================================================

    result["delay_days"] = (
        result["delay_days"]
        .round()
        .astype(int)
    )

    result["duration_overrun_days"] = (
        result["duration_overrun_days"]
        .round()
        .astype(int)
    )

    result["delay_indicator"] = (
        result["delay_indicator"]
        .round(2)
    )

    return result


def get_delay_summary(df):
    """Return summary statistics for delay analysis."""

    if df is None or df.empty:

        return {
            "projects_analyzed": 0,
            "delayed_projects": 0,
            "major_delays": 0,
            "overdue_ongoing_projects": 0,
            "average_delay_days": 0
        }

    analyzed = analyze_delay_anomalies(df)

    delayed = (
        analyzed["delay_days"] > 0
    )

    major_delay = (
        analyzed["delay_days"] > 90
    )

    overdue_ongoing = (
        analyzed["is_ongoing_project"]
        &
        (
            analyzed["delay_days"] > 0
        )
    )

    return {
        "projects_analyzed":
            int(len(analyzed)),

        "delayed_projects":
            int(delayed.sum()),

        "major_delays":
            int(major_delay.sum()),

        "overdue_ongoing_projects":
            int(overdue_ongoing.sum()),

        "average_delay_days":
            round(
                float(
                    analyzed.loc[
                        delayed,
                        "delay_days"
                    ].mean()
                    if delayed.any()
                    else 0
                ),
                2
            )
    }


if __name__ == "__main__":

    test_data = pd.DataFrame({

        "project_id": [
            "TEST001",
            "TEST002",
            "TEST003"
        ],

        "start_date": [
            "2025-01-01",
            "2025-01-01",
            "2025-01-01"
        ],

        "expected_completion_date": [
            "2025-06-30",
            "2025-06-30",
            "2025-06-30"
        ],

        "completion_date": [
            "2025-07-15",
            "2026-01-15",
            "2025-06-20"
        ],

        "project_status": [
            "Completed",
            "Completed",
            "Completed"
        ]
    })

    result = analyze_delay_anomalies(
        test_data
    )

    print("\nDelay detection test:\n")

    print(
        result[
            [
                "project_id",
                "delay_days",
                "duration_overrun_days",
                "delay_indicator",
                "delay_severity",
                "delay_reasons"
            ]
        ].to_string(index=False)
    )

    print("\nSummary:")

    print(
        get_delay_summary(
            test_data
        )
    )