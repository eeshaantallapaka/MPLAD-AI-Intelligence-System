import numpy as np
import pandas as pd


# ============================================================
# MPLAD FEATURE ENGINEERING
# ============================================================
#
# Converts raw MPLAD project data into explainable numerical
# features for the AI anomaly detection pipeline.
#
# IMPORTANT:
# "_demo_injected_issue" is NEVER used.
# The AI must independently discover unusual patterns.
# ============================================================


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def safe_numeric(series):
    """
    Convert values to numeric.
    Invalid values become NaN.
    """
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def safe_datetime(series):
    """
    Convert values to datetime.
    Invalid dates become NaT.
    """
    return pd.to_datetime(
        series,
        errors="coerce"
    )


def safe_divide(numerator, denominator):
    """
    Safely divide pandas Series, NumPy values, or scalar values.

    Handles:
        Series / Series
        Series / scalar
        scalar / Series
        scalar / scalar

    Invalid divisions return 0.
    """

    numerator = pd.to_numeric(
        numerator,
        errors="coerce"
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Series/DataFrame denominator
    # --------------------------------------------------------

    if hasattr(denominator, "replace"):

        denominator = denominator.replace(
            0,
            np.nan
        )

    # --------------------------------------------------------
    # Scalar denominator
    # --------------------------------------------------------

    else:

        try:
            if denominator == 0:
                denominator = np.nan
        except Exception:
            denominator = np.nan

    result = numerator / denominator

    # --------------------------------------------------------
    # Series result
    # --------------------------------------------------------

    if hasattr(result, "replace"):

        return (
            result
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

    # --------------------------------------------------------
    # Scalar result
    # --------------------------------------------------------

    try:

        if pd.isna(result) or np.isinf(result):
            return 0.0

    except Exception:
        return 0.0

    return result


def robust_z_score(series):
    """
    Calculate a robust z-score using median and MAD.

    Robust statistics are less sensitive to extreme outliers.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    median = values.median()

    valid_values = values.dropna()

    if valid_values.empty:
        return pd.Series(
            0.0,
            index=series.index
        )

    mad = np.median(
        np.abs(
            valid_values - median
        )
    )

    if pd.isna(mad) or mad == 0:

        return pd.Series(
            0.0,
            index=series.index
        )

    result = (
        0.6745 *
        (values - median) /
        mad
    )

    return (
        result
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# ------------------------------------------------------------
# Main feature engineering
# ------------------------------------------------------------

def engineer_features(df):
    """
    Generate explainable features from MPLAD project data.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw or cleaned MPLAD project dataset.

    Returns
    -------
    pandas.DataFrame
        Feature-engineered dataset.
    """

    if df is None:

        raise ValueError(
            "Input DataFrame cannot be None."
        )

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        return df.copy()

    # Work on a copy.
    data = df.copy()

    # ========================================================
    # NUMERIC COLUMNS
    # ========================================================

    numeric_columns = [
        "sanctioned_amount",
        "released_amount",
        "expenditure",
        "latitude",
        "longitude",
        "beneficiary_count"
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = safe_numeric(
                data[column]
            )

        else:

            data[column] = 0.0

    # ========================================================
    # DATE COLUMNS
    # ========================================================

    date_columns = [
        "sanction_date",
        "start_date",
        "completion_date"
    ]

    for column in date_columns:

        if column in data.columns:

            data[column] = safe_datetime(
                data[column]
            )

        else:

            data[column] = pd.NaT

    # ========================================================
    # TEXT COLUMNS
    # ========================================================

    text_columns = [
        "project_id",
        "state",
        "district",
        "constituency",
        "MP_name",
        "project_name",
        "project_category",
        "implementing_agency",
        "contractor",
        "project_status"
    ]

    for column in text_columns:

        if column not in data.columns:

            data[column] = ""

        data[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # ========================================================
    # FINANCIAL FEATURES
    # ========================================================

    data["expenditure_ratio"] = safe_divide(
        data["expenditure"],
        data["sanctioned_amount"]
    )

    data["release_ratio"] = safe_divide(
        data["released_amount"],
        data["sanctioned_amount"]
    )

    data["utilization_ratio"] = safe_divide(
        data["expenditure"],
        data["released_amount"]
    )

    data["amount_remaining"] = (
        data["sanctioned_amount"]
        - data["expenditure"]
    )

    data["unspent_amount"] = (
        data["amount_remaining"]
        .clip(lower=0)
    )

    data["expenditure_over_sanction"] = (
        data["expenditure"]
        - data["sanctioned_amount"]
    ).clip(lower=0)

    # ========================================================
    # FINANCIAL WARNING INDICATORS
    # ========================================================

    data["expenditure_exceeds_sanction"] = (
        data["expenditure"]
        >
        data["sanctioned_amount"]
    ).astype(int)

    data["release_exceeds_sanction"] = (
        data["released_amount"]
        >
        data["sanctioned_amount"]
    ).astype(int)

    data["zero_expenditure"] = (
        data["expenditure"] <= 0
    ).astype(int)

    data["high_utilization"] = (
        data["utilization_ratio"] > 0.95
    ).astype(int)

    data["low_utilization"] = (
        data["utilization_ratio"] < 0.25
    ).astype(int)

    # ========================================================
    # PROJECT TIMELINE
    # ========================================================

    data["project_duration_days"] = (
        data["completion_date"]
        - data["start_date"]
    ).dt.days

    today = pd.Timestamp.today().normalize()

    effective_end_date = (
        data["completion_date"]
        .fillna(today)
    )

    data["elapsed_days"] = (
        effective_end_date
        - data["start_date"]
    ).dt.days

    data["elapsed_days"] = (
        data["elapsed_days"]
        .fillna(0)
        .clip(lower=0)
    )

    # ========================================================
    # EXPECTED DURATION
    # ========================================================

    expected_duration = 180

    data["expected_duration_days"] = (
        expected_duration
    )

    data["delay_days"] = (
        data["elapsed_days"]
        - expected_duration
    ).clip(lower=0)

    data["delay_ratio"] = safe_divide(
        data["delay_days"],
        expected_duration
    )

    data["major_delay"] = (
        data["delay_days"] > 180
    ).astype(int)

    # ========================================================
    # DATE QUALITY
    # ========================================================

    data["missing_sanction_date"] = (
        data["sanction_date"].isna()
    ).astype(int)

    data["missing_start_date"] = (
        data["start_date"].isna()
    ).astype(int)

    data["missing_completion_date"] = (
        data["completion_date"].isna()
    ).astype(int)

    data["invalid_timeline"] = (
        (
            data["start_date"].notna()
            &
            data["sanction_date"].notna()
            &
            (
                data["start_date"]
                <
                data["sanction_date"]
            )
        )
        |
        (
            data["completion_date"].notna()
            &
            data["start_date"].notna()
            &
            (
                data["completion_date"]
                <
                data["start_date"]
            )
        )
    ).astype(int)

    # ========================================================
    # BENEFICIARY FEATURES
    # ========================================================

    data["negative_beneficiary_count"] = (
        data["beneficiary_count"] < 0
    ).astype(int)

    data["beneficiary_count"] = (
        data["beneficiary_count"]
        .fillna(0)
    )

    data["beneficiary_per_rupee"] = safe_divide(
        data["beneficiary_count"],
        data["expenditure"]
    )

    data["cost_per_beneficiary"] = safe_divide(
        data["expenditure"],
        data["beneficiary_count"].clip(
            lower=1
        )
    )

    data["cost_per_beneficiary"] = (
        data["cost_per_beneficiary"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    quality_columns = [
        "state",
        "district",
        "constituency",
        "MP_name",
        "project_name",
        "project_category",
        "implementing_agency",
        "contractor",
        "project_status"
    ]

    data["missing_field_count"] = 0

    for column in quality_columns:

        data["missing_field_count"] += (
            data[column] == ""
        ).astype(int)

    data["missing_financial_data"] = (
        data["sanctioned_amount"].isna()
        |
        data["released_amount"].isna()
        |
        data["expenditure"].isna()
    ).astype(int)

    data["data_quality_score"] = (
        data["missing_field_count"]
        +
        data["missing_sanction_date"]
        +
        data["missing_start_date"]
        +
        data["missing_completion_date"]
        +
        data["negative_beneficiary_count"]
        +
        data["invalid_timeline"]
    )

    # ========================================================
    # CONTRACTOR ANALYSIS FEATURES
    # ========================================================

    contractor_counts = (
        data["contractor"]
        .replace("", np.nan)
        .value_counts()
    )

    data["contractor_project_count"] = (
        data["contractor"]
        .map(contractor_counts)
        .fillna(0)
    )

    total_projects = len(data)

    if total_projects > 0:

        data["contractor_project_share"] = (
            data["contractor_project_count"]
            /
            total_projects
        )

    else:

        data["contractor_project_share"] = 0.0

    data["high_contractor_concentration"] = (
        data["contractor_project_share"]
        >= 0.30
    ).astype(int)

    # ========================================================
    # DISTRICT STATISTICS
    # ========================================================

    district_avg = (
        data.groupby("district")[
            "sanctioned_amount"
        ]
        .transform("mean")
    )

    district_median = (
        data.groupby("district")[
            "sanctioned_amount"
        ]
        .transform("median")
    )

    district_exp_avg = (
        data.groupby("district")[
            "expenditure"
        ]
        .transform("mean")
    )

    overall_cost_median = (
        data["sanctioned_amount"]
        .median()
    )

    if pd.isna(overall_cost_median):
        overall_cost_median = 0.0

    overall_exp_median = (
        data["expenditure"]
        .median()
    )

    if pd.isna(overall_exp_median):
        overall_exp_median = 0.0

    data["district_average_cost"] = (
        district_avg
        .fillna(overall_cost_median)
    )

    data["district_median_cost"] = (
        district_median
        .fillna(overall_cost_median)
    )

    data["district_average_expenditure"] = (
        district_exp_avg
        .fillna(overall_exp_median)
    )

    data["deviation_from_district_average"] = (
        safe_divide(
            data["sanctioned_amount"]
            -
            data["district_average_cost"],
            data["district_average_cost"]
        )
    )

    data["district_cost_zscore"] = (
        robust_z_score(
            data["sanctioned_amount"]
            -
            data["district_average_cost"]
        )
    )

    # ========================================================
    # CATEGORY STATISTICS
    # ========================================================

    category_avg = (
        data.groupby("project_category")[
            "sanctioned_amount"
        ]
        .transform("mean")
    )

    category_median = (
        data.groupby("project_category")[
            "sanctioned_amount"
        ]
        .transform("median")
    )

    category_exp_avg = (
        data.groupby("project_category")[
            "expenditure"
        ]
        .transform("mean")
    )

    data["category_average_cost"] = (
        category_avg
        .fillna(overall_cost_median)
    )

    data["category_median_cost"] = (
        category_median
        .fillna(overall_cost_median)
    )

    data["category_average_expenditure"] = (
        category_exp_avg
        .fillna(overall_exp_median)
    )

    data["deviation_from_category_average"] = (
        safe_divide(
            data["sanctioned_amount"]
            -
            data["category_average_cost"],
            data["category_average_cost"]
        )
    )

    data["category_cost_zscore"] = (
        robust_z_score(
            data["sanctioned_amount"]
            -
            data["category_average_cost"]
        )
    )

    # ========================================================
    # STATE STATISTICS
    # ========================================================

    state_avg = (
        data.groupby("state")[
            "sanctioned_amount"
        ]
        .transform("mean")
    )

    state_exp_avg = (
        data.groupby("state")[
            "expenditure"
        ]
        .transform("mean")
    )

    data["state_average_cost"] = (
        state_avg
        .fillna(overall_cost_median)
    )

    data["state_average_expenditure"] = (
        state_exp_avg
        .fillna(overall_exp_median)
    )

    data["deviation_from_state_average"] = (
        safe_divide(
            data["sanctioned_amount"]
            -
            data["state_average_cost"],
            data["state_average_cost"]
        )
    )

    # ========================================================
    # CONSTITUENCY STATISTICS
    # ========================================================

    constituency_avg = (
        data.groupby("constituency")[
            "sanctioned_amount"
        ]
        .transform("mean")
    )

    data["constituency_average_cost"] = (
        constituency_avg
        .fillna(overall_cost_median)
    )

    data["deviation_from_constituency_average"] = (
        safe_divide(
            data["sanctioned_amount"]
            -
            data["constituency_average_cost"],
            data["constituency_average_cost"]
        )
    )

    # ========================================================
    # YEAR STATISTICS
    # ========================================================

    data["sanction_year"] = (
        data["sanction_date"]
        .dt.year
        .fillna(0)
        .astype(int)
    )

    yearly_avg = (
        data.groupby("sanction_year")[
            "sanctioned_amount"
        ]
        .transform("mean")
    )

    data["yearly_average_cost"] = (
        yearly_avg
        .fillna(overall_cost_median)
    )

    data["deviation_from_yearly_average"] = (
        safe_divide(
            data["sanctioned_amount"]
            -
            data["yearly_average_cost"],
            data["yearly_average_cost"]
        )
    )

    # ========================================================
    # GEOGRAPHIC FEATURES
    # ========================================================

    data["missing_coordinates"] = (
        data["latitude"].isna()
        |
        data["longitude"].isna()
    ).astype(int)

    data["invalid_coordinates"] = (
        (
            data["latitude"].notna()
            &
            (
                (data["latitude"] < -90)
                |
                (data["latitude"] > 90)
            )
        )
        |
        (
            data["longitude"].notna()
            &
            (
                (data["longitude"] < -180)
                |
                (data["longitude"] > 180)
            )
        )
    ).astype(int)

    # ========================================================
    # DISTRICT PROJECT DENSITY
    # ========================================================

    district_counts = (
        data["district"]
        .replace("", np.nan)
        .value_counts()
    )

    data["district_project_count"] = (
        data["district"]
        .map(district_counts)
        .fillna(0)
    )

    # ========================================================
    # CONSTITUENCY PROJECT DENSITY
    # ========================================================

    constituency_counts = (
        data["constituency"]
        .replace("", np.nan)
        .value_counts()
    )

    data["constituency_project_count"] = (
        data["constituency"]
        .map(constituency_counts)
        .fillna(0)
    )

    # ========================================================
    # COST ANOMALY INDICATORS
    # ========================================================

    data["extreme_cost_indicator"] = (
        (
            data["deviation_from_category_average"]
            > 2.0
        )
        |
        (
            data["deviation_from_district_average"]
            > 2.0
        )
    ).astype(int)

    data["very_high_cost_indicator"] = (
        (
            data["sanctioned_amount"]
            >
            data["category_average_cost"] * 2
        )
        |
        (
            data["sanctioned_amount"]
            >
            data["district_average_cost"] * 2
        )
    ).astype(int)

    # ========================================================
    # FINANCIAL WARNING SCORE
    # ========================================================

    data["financial_warning_count"] = (
        data["expenditure_exceeds_sanction"]
        +
        data["release_exceeds_sanction"]
        +
        data["low_utilization"]
        +
        data["extreme_cost_indicator"]
    )

    # ========================================================
    # DELAY WARNING SCORE
    # ========================================================

    data["delay_warning_count"] = (
        data["major_delay"]
        +
        data["invalid_timeline"]
    )

    # ========================================================
    # NORMALIZED GEOGRAPHIC FEATURES
    # ========================================================

    data["latitude_normalized"] = (
        data["latitude"] / 90
    )

    data["longitude_normalized"] = (
        data["longitude"] / 180
    )

    data["latitude_normalized"] = (
        data["latitude_normalized"]
        .fillna(0)
    )

    data["longitude_normalized"] = (
        data["longitude_normalized"]
        .fillna(0)
    )

    # ========================================================
    # FINAL NUMERICAL CLEANUP
    # ========================================================

    numerical_columns = (
        data
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    data[numerical_columns] = (
        data[numerical_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    return data


# ============================================================
# GET ML FEATURES
# ============================================================

def get_ml_features(df):
    """
    Return numerical features suitable for Isolation Forest.

    Identifier fields and labels are excluded.
    """

    if df is None or df.empty:

        return pd.DataFrame()

    excluded_features = {
        "id",
        "anomaly_score",
        "risk_score",
        "sanction_year",
        "isolation_prediction",
        "is_anomaly",
        "anomaly_rank"
    }

    # Never allow the synthetic label into the model.
    excluded_features.add(
        "_demo_injected_issue"
    )

    numerical_df = (
        df
        .select_dtypes(
            include=[np.number]
        )
        .copy()
    )

    columns_to_drop = [
        column
        for column in numerical_df.columns
        if column.lower()
        in excluded_features
    ]

    numerical_df = (
        numerical_df
        .drop(
            columns=columns_to_drop,
            errors="ignore"
        )
    )

    # Remove constant columns.
    nunique = (
        numerical_df
        .nunique()
    )

    constant_columns = (
        nunique[
            nunique <= 1
        ]
        .index
        .tolist()
    )

    numerical_df = (
        numerical_df
        .drop(
            columns=constant_columns,
            errors="ignore"
        )
    )

    # Final safety cleanup.
    numerical_df = (
        numerical_df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    return numerical_df


# ============================================================
# FEATURE SUMMARY
# ============================================================

def get_feature_summary(df):
    """
    Return a human-readable feature summary.
    """

    if df is None or df.empty:

        return {
            "rows": 0,
            "features": 0,
            "feature_names": []
        }

    ml_features = get_ml_features(
        df
    )

    return {
        "rows": len(df),
        "features": len(
            ml_features.columns
        ),
        "feature_names": (
            ml_features
            .columns
            .tolist()
        )
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MPLAD FEATURE ENGINEERING MODULE")
    print("=" * 70)
    print()

    print("Functions available:")
    print()
    print("  engineer_features(df)")
    print("  get_ml_features(df)")
    print("  get_feature_summary(df)")
    print()

    print("Feature categories:")
    print()
    print("  Financial indicators")
    print("  Utilization indicators")
    print("  Timeline and delay indicators")
    print("  Beneficiary indicators")
    print("  Contractor indicators")
    print("  District statistics")
    print("  Category statistics")
    print("  State statistics")
    print("  Constituency statistics")
    print("  Geographic indicators")
    print("  Data quality indicators")
    print()

    print("IMPORTANT:")
    print(
        "  _demo_injected_issue is never used by the AI."
    )
    print()