import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import Config
from ml.feature_engineering import (
    engineer_features,
    get_ml_features
)


# ============================================================
# MPLAD AI ANOMALY DETECTOR
# ============================================================
#
# Primary machine-learning model:
#       Isolation Forest
#
# Purpose:
#       Identify projects whose overall feature patterns are
#       unusual compared with the rest of the dataset.
#
# IMPORTANT:
#       The model NEVER uses:
#
#           _demo_injected_issue
#
#       That field exists only inside the synthetic-data
#       generator and must never influence the AI.
#
# AI output is an INDICATOR for human review.
# It does NOT automatically prove fraud.
# ============================================================


MODEL_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "instance",
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIRECTORY,
    "isolation_forest.joblib"
)


# ------------------------------------------------------------
# Ensure model directory exists
# ------------------------------------------------------------

def ensure_model_directory():
    """
    Create the model directory if it does not exist.
    """

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True
    )


# ------------------------------------------------------------
# Build Isolation Forest
# ------------------------------------------------------------

def build_model():
    """
    Create and return the Isolation Forest model.

    Configuration is loaded from config.py.
    """

    return IsolationForest(
        n_estimators=Config.ISOLATION_FOREST_ESTIMATORS,
        contamination=Config.ISOLATION_FOREST_CONTAMINATION,
        random_state=Config.ISOLATION_FOREST_RANDOM_STATE,
        n_jobs=-1
    )


# ------------------------------------------------------------
# Prepare ML matrix
# ------------------------------------------------------------

def prepare_ml_matrix(df):
    """
    Prepare the numerical feature matrix used by Isolation
    Forest.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature-engineered DataFrame.

    Returns
    -------
    pandas.DataFrame
        Clean numerical ML matrix.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    ml_features = get_ml_features(df)

    if ml_features.empty:
        return pd.DataFrame()

    # Replace infinite values.
    ml_features = ml_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Replace missing values using column medians.
    for column in ml_features.columns:

        if ml_features[column].isna().all():
            ml_features[column] = 0.0
        else:
            median_value = ml_features[column].median()

            if pd.isna(median_value):
                median_value = 0.0

            ml_features[column] = (
                ml_features[column]
                .fillna(median_value)
            )

    # Final safety check.
    ml_features = ml_features.fillna(0)

    return ml_features


# ------------------------------------------------------------
# Convert Isolation Forest score into 0–100 anomaly score
# ------------------------------------------------------------

def calculate_anomaly_score(decision_scores):
    """
    Convert Isolation Forest decision_function values into an
    intuitive 0–100 anomaly score.

    Isolation Forest:
        higher decision_function = more normal
        lower decision_function  = more anomalous

    Therefore the score is inverted.

    Returns
    -------
    numpy.ndarray
        Anomaly scores between 0 and 100.
    """

    scores = np.asarray(
        decision_scores,
        dtype=float
    )

    if len(scores) == 0:
        return np.array([])

    # Handle completely identical scores.
    if np.nanmax(scores) == np.nanmin(scores):
        return np.full(
            len(scores),
            50.0
        )

    # Convert normality score into anomaly score.
    min_score = np.nanmin(scores)
    max_score = np.nanmax(scores)

    normalized = (
        (scores - min_score) /
        (max_score - min_score)
    )

    anomaly_scores = (
        1.0 - normalized
    ) * 100.0

    return np.clip(
        anomaly_scores,
        0,
        100
    )


# ------------------------------------------------------------
# Percentile-based anomaly score
# ------------------------------------------------------------

def calculate_percentile_anomaly_score(decision_scores):
    """
    Calculate a percentile-based anomaly score.

    This is useful because it gives a relative ranking of
    suspicious projects.

    Example:
        95 means the project is among the most unusual
        observations in the current dataset.
    """

    scores = pd.Series(
        np.asarray(
            decision_scores,
            dtype=float
        )
    )

    if scores.empty:
        return np.array([])

    percentile = scores.rank(
        method="average",
        pct=True
    )

    anomaly_score = (
        1.0 - percentile
    ) * 100.0

    return anomaly_score.to_numpy()


# ------------------------------------------------------------
# Detect anomalies
# ------------------------------------------------------------

def detect_anomalies(
    df,
    save_model=True
):
    """
    Run Isolation Forest anomaly detection.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw or cleaned MPLAD dataset.

    save_model : bool
        Whether to save the trained model to disk.

    Returns
    -------
    tuple
        (
            feature_engineered_dataframe,
            metadata
        )
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
        return df.copy(), {
            "success": False,
            "message": "Dataset is empty.",
            "projects": 0,
            "anomalies": 0
        }

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    feature_df = engineer_features(df)

    # --------------------------------------------------------
    # Prepare numerical ML features
    # --------------------------------------------------------

    X = prepare_ml_matrix(
        feature_df
    )

    if X.empty:
        return feature_df, {
            "success": False,
            "message": "No usable numerical features found.",
            "projects": len(feature_df),
            "anomalies": 0
        }

    # --------------------------------------------------------
    # Train Isolation Forest
    # --------------------------------------------------------

    model = build_model()

    model.fit(X)

    # --------------------------------------------------------
    # Model predictions
    #
    #   1  = normal
    #  -1  = anomaly
    # --------------------------------------------------------

    predictions = model.predict(X)

    # --------------------------------------------------------
    # Raw decision scores
    #
    # Higher = more normal
    # Lower  = more anomalous
    # --------------------------------------------------------

    decision_scores = model.decision_function(X)

    # --------------------------------------------------------
    # Convert to intuitive 0–100 score
    # --------------------------------------------------------

    normalized_scores = calculate_anomaly_score(
        decision_scores
    )

    percentile_scores = calculate_percentile_anomaly_score(
        decision_scores
    )

    # --------------------------------------------------------
    # Use the percentile score as the main anomaly score.
    #
    # This provides a stable ranking:
    #
    #   0   = relatively normal
    #   100 = extremely unusual
    # --------------------------------------------------------

    feature_df["anomaly_score"] = (
        percentile_scores
    )

    feature_df["isolation_forest_score"] = (
        normalized_scores
    )

    feature_df["isolation_decision_score"] = (
        decision_scores
    )

    feature_df["isolation_prediction"] = (
        predictions
    )

    feature_df["is_anomaly"] = (
        predictions == -1
    ).astype(int)

    # --------------------------------------------------------
    # Anomaly severity
    # --------------------------------------------------------

    def severity(score):

        if score >= 90:
            return "CRITICAL"

        if score >= 75:
            return "HIGH"

        if score >= 50:
            return "MEDIUM"

        if score >= 25:
            return "LOW"

        return "NORMAL"

    feature_df["anomaly_severity"] = (
        feature_df["anomaly_score"]
        .apply(severity)
    )

    # --------------------------------------------------------
    # Rank projects by anomaly score
    # --------------------------------------------------------

    feature_df["anomaly_rank"] = (
        feature_df["anomaly_score"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    if save_model:

        ensure_model_directory()

        model_package = {
            "model": model,
            "feature_names": X.columns.tolist(),
            "model_type": "IsolationForest",
            "estimators": Config.ISOLATION_FOREST_ESTIMATORS,
            "random_state": Config.ISOLATION_FOREST_RANDOM_STATE
        }

        joblib.dump(
            model_package,
            MODEL_PATH
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    anomaly_count = int(
        (predictions == -1).sum()
    )

    normal_count = int(
        (predictions == 1).sum()
    )

    metadata = {
        "success": True,
        "message": "Isolation Forest anomaly detection completed.",
        "projects": len(feature_df),
        "anomalies": anomaly_count,
        "normal_projects": normal_count,
        "feature_count": len(X.columns),
        "feature_names": X.columns.tolist(),
        "model_type": "Isolation Forest",
        "model_path": MODEL_PATH if save_model else None
    }

    return feature_df, metadata


# ------------------------------------------------------------
# Load saved model
# ------------------------------------------------------------

def load_saved_model():
    """
    Load the previously trained Isolation Forest model.

    Returns
    -------
    dict
        Saved model package.

    Raises
    ------
    FileNotFoundError
        If the model has not been trained yet.
    """

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Isolation Forest model has not been trained yet."
        )

    return joblib.load(
        MODEL_PATH
    )


# ------------------------------------------------------------
# Predict using saved model
# ------------------------------------------------------------

def predict_with_saved_model(df):
    """
    Run anomaly prediction using an already-trained model.

    This is useful later when processing new uploaded datasets.
    """

    if df is None:
        raise ValueError(
            "Input DataFrame cannot be None."
        )

    if df.empty:
        return df.copy()

    # Load saved model.
    package = load_saved_model()

    model = package["model"]

    expected_features = package[
        "feature_names"
    ]

    # Feature engineering.
    feature_df = engineer_features(df)

    # Prepare numerical matrix.
    X = prepare_ml_matrix(
        feature_df
    )

    # Make sure the same features used during training exist.
    for column in expected_features:

        if column not in X.columns:
            X[column] = 0.0

    # Remove unexpected features.
    X = X[
        expected_features
    ]

    # Predict.
    predictions = model.predict(X)

    decision_scores = model.decision_function(X)

    anomaly_scores = calculate_anomaly_score(
        decision_scores
    )

    percentile_scores = calculate_percentile_anomaly_score(
        decision_scores
    )

    feature_df["anomaly_score"] = (
        percentile_scores
    )

    feature_df["isolation_forest_score"] = (
        anomaly_scores
    )

    feature_df["isolation_decision_score"] = (
        decision_scores
    )

    feature_df["isolation_prediction"] = (
        predictions
    )

    feature_df["is_anomaly"] = (
        predictions == -1
    ).astype(int)

    return feature_df


# ------------------------------------------------------------
# Get top anomalies
# ------------------------------------------------------------

def get_top_anomalies(
    df,
    limit=20
):
    """
    Return the highest-scoring suspicious projects.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if "anomaly_score" not in df.columns:
        return pd.DataFrame()

    columns = [
        column
        for column in [
            "project_id",
            "state",
            "district",
            "constituency",
            "project_name",
            "project_category",
            "sanctioned_amount",
            "expenditure",
            "contractor",
            "anomaly_score",
            "anomaly_severity",
            "is_anomaly"
        ]
        if column in df.columns
    ]

    return (
        df.sort_values(
            "anomaly_score",
            ascending=False
        )
        [columns]
        .head(limit)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------
# Get anomaly distribution
# ------------------------------------------------------------

def get_anomaly_distribution(df):
    """
    Return the distribution of anomaly severity levels.
    """

    if df is None or df.empty:
        return {}

    if "anomaly_severity" not in df.columns:
        return {}

    distribution = (
        df["anomaly_severity"]
        .value_counts()
        .to_dict()
    )

    return {
        str(key): int(value)
        for key, value in distribution.items()
    }


# ------------------------------------------------------------
# Module test
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("MPLAD AI Anomaly Detection Module")
    print("=" * 70)
    print()

    print("Algorithm:")
    print("  Isolation Forest")
    print()

    print("Purpose:")
    print("  Detect statistically unusual project patterns.")
    print()

    print("Model configuration:")
    print(
        f"  Estimators: "
        f"{Config.ISOLATION_FOREST_ESTIMATORS}"
    )

    print(
        f"  Contamination: "
        f"{Config.ISOLATION_FOREST_CONTAMINATION}"
    )

    print(
        f"  Random state: "
        f"{Config.ISOLATION_FOREST_RANDOM_STATE}"
    )

    print()

    print("Available functions:")
    print("  - detect_anomalies(df)")
    print("  - predict_with_saved_model(df)")
    print("  - load_saved_model()")
    print("  - get_top_anomalies(df)")
    print("  - get_anomaly_distribution(df)")
    print()

    print("IMPORTANT:")
    print("  _demo_injected_issue is never used by the AI.")
    print()