import re
from difflib import SequenceMatcher

import pandas as pd
import numpy as np


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text so similar project descriptions can be compared.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = str(value).lower().strip()

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TEXT SIMILARITY
# ============================================================

def text_similarity(value_a, value_b):
    """
    Calculate similarity between two text values.
    """

    a = normalize_text(value_a)
    b = normalize_text(value_b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# NUMERIC SIMILARITY
# ============================================================

def numeric_similarity(value_a, value_b):
    """
    Compare two numeric values.
    Returns a value between 0 and 1.
    """

    try:

        a = float(value_a)
        b = float(value_b)

    except (TypeError, ValueError):

        return 0.0

    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0

    if a == 0 and b == 0:
        return 1.0

    denominator = max(
        abs(a),
        abs(b),
        1.0
    )

    difference = abs(a - b)

    similarity = 1.0 - (
        difference / denominator
    )

    return max(
        0.0,
        min(
            1.0,
            similarity
        )
    )


# ============================================================
# LOCATION SIMILARITY
# ============================================================

def location_similarity(row_a, row_b):
    """
    Compare state, district and constituency.
    """

    state_score = text_similarity(
        row_a.get("state"),
        row_b.get("state")
    )

    district_score = text_similarity(
        row_a.get("district"),
        row_b.get("district")
    )

    constituency_score = text_similarity(
        row_a.get("constituency"),
        row_b.get("constituency")
    )

    return (
        state_score * 0.25
        +
        district_score * 0.45
        +
        constituency_score * 0.30
    )


# ============================================================
# PROJECT SIMILARITY
# ============================================================

def calculate_project_similarity(
    row_a,
    row_b
):
    """
    Calculate overall similarity between two projects.
    """

    project_name_score = text_similarity(
        row_a.get("project_name"),
        row_b.get("project_name")
    )

    category_score = text_similarity(
        row_a.get("project_category"),
        row_b.get("project_category")
    )

    contractor_score = text_similarity(
        row_a.get("contractor"),
        row_b.get("contractor")
    )

    location_score = location_similarity(
        row_a,
        row_b
    )

    amount_score = numeric_similarity(
        row_a.get("sanctioned_amount"),
        row_b.get("sanctioned_amount")
    )

    # --------------------------------------------------------
    # Weighted similarity
    # --------------------------------------------------------

    score = (
        project_name_score * 0.35
        +
        location_score * 0.30
        +
        category_score * 0.10
        +
        contractor_score * 0.10
        +
        amount_score * 0.15
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                score
            )
        )
    )


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def detect_duplicate_projects(
    dataframe,
    similarity_threshold=0.85
):
    """
    Detect potentially duplicate or highly similar projects.

    This is an indicator for human review.
    It does NOT establish that projects are fraudulent.
    """

    if dataframe is None:

        return pd.DataFrame()

    if dataframe.empty:

        result = dataframe.copy()

        result["duplicate_indicator"] = 0.0
        result["duplicate_match"] = ""
        result["duplicate_similarity"] = 0.0
        result["duplicate_reason"] = ""

        return result


    df = dataframe.copy()

    # --------------------------------------------------------
    # Ensure expected columns exist
    # --------------------------------------------------------

    required_columns = [
        "project_id",
        "project_name",
        "state",
        "district",
        "constituency",
        "project_category",
        "sanctioned_amount",
        "contractor"
    ]

    for column in required_columns:

        if column not in df.columns:

            df[column] = ""


    # --------------------------------------------------------
    # Numeric cleanup
    # --------------------------------------------------------

    df["sanctioned_amount"] = pd.to_numeric(
        df["sanctioned_amount"],
        errors="coerce"
    ).fillna(0)


    # --------------------------------------------------------
    # Prepare result columns
    # --------------------------------------------------------

    df["duplicate_indicator"] = 0.0

    df["duplicate_match"] = ""

    df["duplicate_similarity"] = 0.0

    df["duplicate_reason"] = ""


    # --------------------------------------------------------
    # Avoid comparing every row with every other row
    # --------------------------------------------------------
    # We first group projects by broad location.
    # This keeps the algorithm practical for larger datasets.

    grouping_columns = [
        "state",
        "district"
    ]

    normalized_groups = []

    for _, row in df.iterrows():

        state = normalize_text(
            row.get("state")
        )

        district = normalize_text(
            row.get("district")
        )

        normalized_groups.append(
            (
                state,
                district
            )
        )

    df["_duplicate_group"] = normalized_groups


    # --------------------------------------------------------
    # Compare projects within geographic groups
    # --------------------------------------------------------

    for _, group in df.groupby(
        "_duplicate_group",
        sort=False
    ):

        indexes = list(group.index)

        if len(indexes) < 2:
            continue


        for position_a in range(
            len(indexes)
        ):

            index_a = indexes[position_a]

            row_a = df.loc[index_a]


            for position_b in range(
                position_a + 1,
                len(indexes)
            ):

                index_b = indexes[position_b]

                row_b = df.loc[index_b]


                # ------------------------------------------------
                # Never compare the same project
                # ------------------------------------------------

                id_a = normalize_text(
                    row_a.get("project_id")
                )

                id_b = normalize_text(
                    row_b.get("project_id")
                )

                if id_a and id_b and id_a == id_b:
                    continue


                similarity = calculate_project_similarity(
                    row_a,
                    row_b
                )


                if similarity < similarity_threshold:
                    continue


                # ------------------------------------------------
                # Keep the stronger match for each project
                # ------------------------------------------------

                current_similarity_a = float(
                    df.at[
                        index_a,
                        "duplicate_similarity"
                    ]
                )

                current_similarity_b = float(
                    df.at[
                        index_b,
                        "duplicate_similarity"
                    ]
                )


                if similarity >= current_similarity_a:

                    df.at[
                        index_a,
                        "duplicate_similarity"
                    ] = round(
                        similarity,
                        4
                    )

                    df.at[
                        index_a,
                        "duplicate_match"
                    ] = str(
                        row_b.get("project_id", "")
                    )


                if similarity >= current_similarity_b:

                    df.at[
                        index_b,
                        "duplicate_similarity"
                    ] = round(
                        similarity,
                        4
                    )

                    df.at[
                        index_b,
                        "duplicate_match"
                    ] = str(
                        row_a.get("project_id", "")
                    )


    # ========================================================
    # CONVERT SIMILARITY TO INDICATOR
    # ========================================================

    matched = (
        df["duplicate_similarity"]
        >= similarity_threshold
    )


    df.loc[
        matched,
        "duplicate_indicator"
    ] = (
        (
            df.loc[
                matched,
                "duplicate_similarity"
            ]
            - similarity_threshold
        )
        /
        max(
            0.01,
            1.0 - similarity_threshold
        )
        * 100
    )


    df["duplicate_indicator"] = (
        df["duplicate_indicator"]
        .clip(
            0,
            100
        )
        .round(2)
    )


    # ========================================================
    # GENERATE EXPLANATIONS
    # ========================================================

    for index in df.index:

        similarity = float(
            df.at[
                index,
                "duplicate_similarity"
            ]
        )

        if similarity < similarity_threshold:
            continue


        match_id = str(
            df.at[
                index,
                "duplicate_match"
            ]
        ).strip()


        if not match_id:
            continue


        reasons = []

        reasons.append(
            f"Highly similar project detected: {match_id}"
        )

        reasons.append(
            f"Similarity score: {similarity * 100:.1f}%"
        )


        project_name = normalize_text(
            df.at[
                index,
                "project_name"
            ]
        )


        category = normalize_text(
            df.at[
                index,
                "project_category"
            ]
        )


        if project_name:
            reasons.append(
                "Project description is similar"
            )


        if category:
            reasons.append(
                "Project category is similar"
            )


        reasons.append(
            "Location is within the same geographic grouping"
        )


        df.at[
            index,
            "duplicate_reason"
        ] = " | ".join(
            reasons
        )


    # --------------------------------------------------------
    # Cleanup temporary column
    # --------------------------------------------------------

    df.drop(
        columns=[
            "_duplicate_group"
        ],
        inplace=True,
        errors="ignore"
    )


    return df


# ============================================================
# SIMPLE SUMMARY
# ============================================================

def summarize_duplicate_detection(
    dataframe
):
    """
    Generate summary statistics for duplicate analysis.
    """

    if dataframe is None or dataframe.empty:

        return {
            "projects_checked": 0,
            "potential_duplicates": 0,
            "average_similarity": 0.0,
            "highest_similarity": 0.0
        }


    indicator = pd.to_numeric(
        dataframe.get(
            "duplicate_indicator",
            0
        ),
        errors="coerce"
    ).fillna(0)


    similarity = pd.to_numeric(
        dataframe.get(
            "duplicate_similarity",
            0
        ),
        errors="coerce"
    ).fillna(0)


    potential_duplicates = int(
        (indicator > 0).sum()
    )


    return {
        "projects_checked": int(
            len(dataframe)
        ),

        "potential_duplicates": potential_duplicates,

        "average_similarity": round(
            float(
                similarity.mean()
            ) * 100,
            2
        ),

        "highest_similarity": round(
            float(
                similarity.max()
            ) * 100,
            2
        )
    }