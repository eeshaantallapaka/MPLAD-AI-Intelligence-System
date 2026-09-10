import pandas as pd
import numpy as np


def _number(value):
    """
    Safely convert a value to a number.
    """
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def analyze_financial_anomalies(df):
    """
    Detect explainable financial anomalies in MPLAD projects.

    This function does NOT declare fraud.
    It identifies financial risk indicators for human review.
    """

    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    result = df.copy()

    # ---------------------------------------------------------
    # Normalize column names
    # ---------------------------------------------------------
    result.columns = [
        str(column).strip().lower()
        for column in result.columns
    ]

    # ---------------------------------------------------------
    # Make sure expected financial columns exist
    # ---------------------------------------------------------
    financial_columns = [
        "sanctioned_amount",
        "estimated_cost",
        "released_amount",
        "amount_released",
        "expenditure",
        "actual_expenditure",
        "amount_utilized",
        "amount_utilised",
    ]

    for column in financial_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            ).fillna(0)

    # ---------------------------------------------------------
    # Resolve different dataset schemas
    # ---------------------------------------------------------

    if "sanctioned_amount" not in result.columns:
        result["sanctioned_amount"] = 0.0

    if (
        result["sanctioned_amount"] == 0
    ).all() and "estimated_cost" in result.columns:
        result["sanctioned_amount"] = result["estimated_cost"]

    if "released_amount" not in result.columns:
        if "amount_released" in result.columns:
            result["released_amount"] = result["amount_released"]
        else:
            result["released_amount"] = 0.0

    if "expenditure" not in result.columns:
        if "actual_expenditure" in result.columns:
            result["expenditure"] = result["actual_expenditure"]
        elif "amount_utilized" in result.columns:
            result["expenditure"] = result["amount_utilized"]
        elif "amount_utilised" in result.columns:
            result["expenditure"] = result["amount_utilised"]
        else:
            result["expenditure"] = 0.0

    # ---------------------------------------------------------
    # Financial calculations
    # ---------------------------------------------------------

    sanctioned = result["sanctioned_amount"]
    released = result["released_amount"]
    expenditure = result["expenditure"]

    # Avoid division by zero
    safe_sanctioned = sanctioned.replace(0, np.nan)

    result["financial_utilization"] = (
        expenditure / safe_sanctioned
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    result["release_ratio"] = (
        released / safe_sanctioned
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    result["cost_overrun_amount"] = (
        expenditure - sanctioned
    )

    result["cost_overrun_ratio"] = (
        expenditure / safe_sanctioned - 1
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---------------------------------------------------------
    # Individual financial indicators
    # ---------------------------------------------------------

    result["financial_indicator"] = 0.0

    # Human-readable reasons
    result["financial_reasons"] = ""

    def add_reason(index, reason):
        existing = str(result.at[index, "financial_reasons"])

        if existing and existing != "nan":
            result.at[index, "financial_reasons"] = (
                existing + " || " + reason
            )
        else:
            result.at[index, "financial_reasons"] = reason

    # ---------------------------------------------------------
    # Rule 1: Expenditure exceeds sanctioned amount
    # ---------------------------------------------------------

    over_sanction = (
        (sanctioned > 0) &
        (expenditure > sanctioned)
    )

    for index in result.index[over_sanction]:
        result.at[index, "financial_indicator"] += 40

        overrun = _number(result.at[index, "cost_overrun_ratio"]) * 100

        add_reason(
            index,
            f"Expenditure exceeds sanctioned amount by "
            f"{overrun:.1f}%."
        )

    # ---------------------------------------------------------
    # Rule 2: Released amount exceeds sanctioned amount
    # ---------------------------------------------------------

    release_over_sanction = (
        (sanctioned > 0) &
        (released > sanctioned)
    )

    for index in result.index[release_over_sanction]:
        result.at[index, "financial_indicator"] += 30

        add_reason(
            index,
            "Released amount exceeds sanctioned amount."
        )

    # ---------------------------------------------------------
    # Rule 3: Very low utilization
    # ---------------------------------------------------------

    low_utilization = (
        (sanctioned > 0) &
        (result["financial_utilization"] < 0.25) &
        (released > 0)
    )

    for index in result.index[low_utilization]:
        result.at[index, "financial_indicator"] += 25

        utilization = (
            _number(result.at[index, "financial_utilization"])
            * 100
        )

        add_reason(
            index,
            f"Low fund utilization ({utilization:.1f}%)."
        )

    # ---------------------------------------------------------
    # Rule 4: Extremely high expenditure
    # ---------------------------------------------------------

    extreme_cost = (
        (sanctioned > 0) &
        (result["financial_utilization"] >= 2.0)
    )

    for index in result.index[extreme_cost]:
        result.at[index, "financial_indicator"] += 30

        add_reason(
            index,
            "Expenditure is at least twice the sanctioned amount."
        )

    # ---------------------------------------------------------
    # Rule 5: Moderate cost overrun
    # ---------------------------------------------------------

    moderate_overrun = (
        (sanctioned > 0) &
        (result["financial_utilization"] >= 1.10) &
        (result["financial_utilization"] < 2.0)
    )

    for index in result.index[moderate_overrun]:
        result.at[index, "financial_indicator"] += 15

        add_reason(
            index,
            "Moderate expenditure overrun detected."
        )

    # ---------------------------------------------------------
    # Rule 6: Zero expenditure despite released funds
    # ---------------------------------------------------------

    zero_expenditure = (
        (released > 0) &
        (expenditure <= 0)
    )

    for index in result.index[zero_expenditure]:
        result.at[index, "financial_indicator"] += 20

        add_reason(
            index,
            "Funds released but no expenditure recorded."
        )

    # ---------------------------------------------------------
    # Rule 7: High utilization
    # ---------------------------------------------------------

    high_utilization = (
        (sanctioned > 0) &
        (result["financial_utilization"] >= 0.95) &
        (result["financial_utilization"] <= 1.10)
    )

    for index in result.index[high_utilization]:
        result.at[index, "financial_indicator"] += 5

    # ---------------------------------------------------------
    # Cap financial indicator
    # ---------------------------------------------------------

    result["financial_indicator"] = (
        result["financial_indicator"]
        .clip(lower=0, upper=100)
    )

    # ---------------------------------------------------------
    # Financial severity
    # ---------------------------------------------------------

    def financial_severity(score):
        if score >= 75:
            return "CRITICAL"
        elif score >= 50:
            return "HIGH"
        elif score >= 25:
            return "MEDIUM"
        elif score > 0:
            return "LOW"
        return "NORMAL"

    result["financial_severity"] = (
        result["financial_indicator"]
        .apply(financial_severity)
    )

    return result


def get_financial_summary(df):
    """
    Return summary statistics for financial anomaly analysis.
    """

    if df is None or df.empty:
        return {
            "projects_analyzed": 0,
            "financial_anomalies": 0,
            "cost_overruns": 0,
            "low_utilization": 0,
            "release_overruns": 0,
            "zero_expenditure": 0,
        }

    analyzed = analyze_financial_anomalies(df)

    return {
        "projects_analyzed": int(len(analyzed)),

        "financial_anomalies": int(
            (analyzed["financial_indicator"] > 0).sum()
        ),

        "cost_overruns": int(
            (
                analyzed["expenditure"]
                > analyzed["sanctioned_amount"]
            ).sum()
        ),

        "low_utilization": int(
            (
                analyzed["financial_utilization"]
                < 0.25
            ).sum()
        ),

        "release_overruns": int(
            (
                analyzed["released_amount"]
                > analyzed["sanctioned_amount"]
            ).sum()
        ),

        "zero_expenditure": int(
            (
                (analyzed["released_amount"] > 0) &
                (analyzed["expenditure"] <= 0)
            ).sum()
        ),
    }


if __name__ == "__main__":
    # Small standalone test
    test_data = pd.DataFrame({
        "project_id": ["TEST001", "TEST002", "TEST003"],
        "sanctioned_amount": [1000000, 1000000, 1000000],
        "released_amount": [900000, 1000000, 1000000],
        "expenditure": [950000, 1500000, 100000],
    })

    result = analyze_financial_anomalies(test_data)

    print("\nFinancial anomaly test:")
    print(
        result[
            [
                "project_id",
                "financial_utilization",
                "financial_indicator",
                "financial_severity",
                "financial_reasons",
            ]
        ].to_string(index=False)
    )

    print("\nSummary:")
    print(get_financial_summary(test_data))