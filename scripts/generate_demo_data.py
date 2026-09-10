import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TOTAL_PROJECTS = 1000

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "sample_mplad_data.csv"
)


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# STATES AND DISTRICTS
# ============================================================

REGIONS = {

    "Andhra Pradesh": [
        "Guntur",
        "Krishna",
        "Nellore",
        "Visakhapatnam"
    ],

    "Telangana": [
        "Hyderabad",
        "Warangal",
        "Nalgonda",
        "Karimnagar"
    ],

    "Karnataka": [
        "Bengaluru Urban",
        "Mysuru",
        "Belagavi",
        "Dharwad"
    ],

    "Tamil Nadu": [
        "Chennai",
        "Coimbatore",
        "Madurai",
        "Salem"
    ],

    "Maharashtra": [
        "Pune",
        "Nagpur",
        "Nashik",
        "Aurangabad"
    ],

    "Uttar Pradesh": [
        "Lucknow",
        "Kanpur Nagar",
        "Agra",
        "Varanasi"
    ],

    "Rajasthan": [
        "Jaipur",
        "Jodhpur",
        "Udaipur",
        "Kota"
    ],

    "West Bengal": [
        "Kolkata",
        "Howrah",
        "Darjeeling",
        "Murshidabad"
    ],

    "Gujarat": [
        "Ahmedabad",
        "Surat",
        "Vadodara",
        "Rajkot"
    ],

    "Odisha": [
        "Khordha",
        "Cuttack",
        "Puri",
        "Ganjam"
    ]
}


# ============================================================
# APPROXIMATE STATE COORDINATES
# ============================================================

STATE_COORDINATES = {

    "Andhra Pradesh":
        (16.5000, 80.6000),

    "Telangana":
        (17.3850, 78.4867),

    "Karnataka":
        (12.9716, 77.5946),

    "Tamil Nadu":
        (13.0827, 80.2707),

    "Maharashtra":
        (19.0760, 72.8777),

    "Uttar Pradesh":
        (26.8467, 80.9462),

    "Rajasthan":
        (26.9124, 75.7873),

    "West Bengal":
        (22.5726, 88.3639),

    "Gujarat":
        (23.0225, 72.5714),

    "Odisha":
        (20.2961, 85.8245)
}


# ============================================================
# MP NAMES
# ============================================================

MP_NAMES = [

    "Amit Kumar",
    "Rajesh Sharma",
    "Suresh Reddy",
    "Priya Singh",
    "Anil Verma",
    "Vijay Kumar",
    "Meena Patel",
    "Arun Rao",
    "Deepak Gupta",
    "Kavita Joshi",
    "Manoj Yadav",
    "Ramesh Naidu",
    "Sunita Das",
    "Rohit Mehta",
    "Naveen Singh"
]


# ============================================================
# PROJECT CATEGORIES
# ============================================================

CATEGORIES = [

    "Roads",
    "Drinking Water",
    "Education",
    "Healthcare",
    "Sanitation",
    "Community Infrastructure",
    "Electricity",
    "Public Facilities",
    "Sports Infrastructure",
    "Irrigation"
]


# ============================================================
# IMPLEMENTING AGENCIES
# ============================================================

AGENCIES = [

    "State Public Works Department",
    "District Rural Development Agency",
    "Municipal Corporation",
    "Zilla Parishad",
    "Rural Development Department",
    "Urban Local Body",
    "District Administration",
    "State Water Resources Department"
]


# ============================================================
# CONTRACTORS
# ============================================================

CONTRACTORS = [

    "Shree Infrastructure Pvt Ltd",
    "Bharat Construction Co",
    "National Projects Ltd",
    "Apex Civil Works",
    "Sunrise Engineering",
    "Unity Infrastructure",
    "Krishna Builders",
    "Prime Development Works",
    "Saksham Projects",
    "Vardhan Construction",
    "Greenfield Contractors",
    "Reliable Infra Solutions",
    "Metro Civil Contractors",
    "Eastern Infrastructure",
    "Dakshin Projects"
]


# ============================================================
# PROJECT NAME TEMPLATES
# ============================================================

PROJECT_TEMPLATES = {

    "Roads": [

        "Construction of CC Road",
        "Improvement of Village Road",
        "Construction of Internal Roads",
        "Road Widening and Improvement",
        "Construction of Approach Road"
    ],

    "Drinking Water": [

        "Installation of Drinking Water Supply System",
        "Construction of Overhead Water Tank",
        "Pipeline Extension for Drinking Water",
        "Rural Drinking Water Facility"
    ],

    "Education": [

        "Construction of Additional Classrooms",
        "School Building Improvement",
        "Construction of School Laboratory",
        "Renovation of Government School"
    ],

    "Healthcare": [

        "Construction of Primary Health Centre",
        "Renovation of Health Sub Centre",
        "Healthcare Facility Improvement",
        "Construction of Community Health Facility"
    ],

    "Sanitation": [

        "Construction of Community Toilet",
        "Village Sanitation Infrastructure",
        "Waste Management Facility",
        "Drainage Improvement Works"
    ],

    "Community Infrastructure": [

        "Construction of Community Hall",
        "Village Community Centre",
        "Construction of Public Utility Building",
        "Community Infrastructure Development"
    ],

    "Electricity": [

        "Installation of Street Lighting",
        "Electricity Infrastructure Improvement",
        "Solar Lighting Installation",
        "Village Electrification Works"
    ],

    "Public Facilities": [

        "Construction of Public Facility",
        "Renovation of Public Building",
        "Construction of Citizen Service Centre",
        "Public Infrastructure Improvement"
    ],

    "Sports Infrastructure": [

        "Construction of Sports Ground",
        "Development of Community Sports Facility",
        "Construction of Indoor Sports Hall",
        "Sports Infrastructure Improvement"
    ],

    "Irrigation": [

        "Construction of Minor Irrigation Facility",
        "Canal Improvement Works",
        "Village Irrigation Infrastructure",
        "Water Conservation Structure"
    ]
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def random_date(
    start_date,
    end_date
):
    """
    Generate a random date between two dates.
    """

    days = (
        end_date -
        start_date
    ).days

    return (
        start_date +
        timedelta(
            days=random.randint(
                0,
                days
            )
        )
    )


def money_value(
    minimum,
    maximum
):
    """
    Generate a realistic financial value.
    """

    value = random.uniform(
        minimum,
        maximum
    )

    return round(
        value / 1000
    ) * 1000


def choose_project_name(
    category
):
    """
    Select a project name based
    on the project category.
    """

    return random.choice(
        PROJECT_TEMPLATES[
            category
        ]
    )


def generate_coordinates(
    state
):
    """
    Generate coordinates around
    an approximate state location.
    """

    base_lat, base_lon = (
        STATE_COORDINATES[
            state
        ]
    )

    latitude = (
        base_lat +
        np.random.normal(
            0,
            0.45
        )
    )

    longitude = (
        base_lon +
        np.random.normal(
            0,
            0.45
        )
    )

    return (
        round(latitude, 6),
        round(longitude, 6)
    )


# ============================================================
# GENERATE ONE NORMAL PROJECT
# ============================================================

def generate_normal_project(
    index
):

    state = random.choice(
        list(REGIONS.keys())
    )

    district = random.choice(
        REGIONS[state]
    )

    constituency_number = random.randint(
        1,
        8
    )

    constituency = (
        f"{district} "
        f"Constituency "
        f"{constituency_number}"
    )

    mp_name = random.choice(
        MP_NAMES
    )

    category = random.choice(
        CATEGORIES
    )

    project_name = choose_project_name(
        category
    )

    sanctioned_amount = money_value(
        500000,
        5000000
    )

    released_amount = round(
        sanctioned_amount *
        random.uniform(
            0.75,
            1.0
        ),
        2
    )

    expenditure = round(
        released_amount *
        random.uniform(
            0.45,
            0.95
        ),
        2
    )

    sanction_date = random_date(
        datetime(
            2021,
            1,
            1
        ),
        datetime(
            2025,
            6,
            30
        )
    )

    start_date = (
        sanction_date +
        timedelta(
            days=random.randint(
                15,
                90
            )
        )
    )

    duration = random.randint(
        60,
        240
    )

    completion_date = (
        start_date +
        timedelta(
            days=duration
        )
    )

    today = datetime(
        2026,
        9,
        1
    )

    if completion_date <= today:

        status = random.choices(

            [
                "Completed",
                "Delayed"
            ],

            weights=[
                0.85,
                0.15
            ]

        )[0]

    else:

        status = random.choice(

            [
                "Ongoing",
                "Under Implementation"
            ]
        )

    contractor = random.choice(
        CONTRACTORS
    )

    agency = random.choice(
        AGENCIES
    )

    latitude, longitude = (
        generate_coordinates(
            state
        )
    )

    beneficiary_count = random.randint(
        100,
        15000
    )

    return {

        "project_id":
            f"MPLAD-2026-{index:05d}",

        "state":
            state,

        "district":
            district,

        "constituency":
            constituency,

        "MP_name":
            mp_name,

        "project_name":
            project_name,

        "project_category":
            category,

        "sanctioned_amount":
            float(sanctioned_amount),

        "released_amount":
            float(released_amount),

        "expenditure":
            float(expenditure),

        "project_status":
            status,

        "sanction_date":
            sanction_date.strftime(
                "%Y-%m-%d"
            ),

        "start_date":
            start_date.strftime(
                "%Y-%m-%d"
            ),

        "completion_date":
            completion_date.strftime(
                "%Y-%m-%d"
            ),

        "implementing_agency":
            agency,

        "contractor":
            contractor,

        "latitude":
            float(latitude),

        "longitude":
            float(longitude),

        "beneficiary_count":
            int(beneficiary_count),

        # IMPORTANT:
        # This field exists only for
        # synthetic-data testing.
        "_demo_injected_issue":
            "NORMAL"
    }


# ============================================================
# MAIN DATASET GENERATOR
# ============================================================

def generate_dataset():

    print()

    print("=" * 70)

    print(
        "MPLAD SYNTHETIC DATASET GENERATOR"
    )

    print(
        "Smart India Hackathon 2026 - SIH26102"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # CREATE NORMAL PROJECTS
    # --------------------------------------------------------

    print(
        "Generating normal MPLAD projects..."
    )

    projects = []

    for index in range(
        1,
        TOTAL_PROJECTS + 1
    ):

        projects.append(
            generate_normal_project(
                index
            )
        )

    df = pd.DataFrame(
        projects
    )

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # Force numeric columns to float.
    # This prevents pandas dtype errors when
    # injecting decimal anomaly values later.
    # --------------------------------------------------------

    numeric_columns = [

        "sanctioned_amount",
        "released_amount",
        "expenditure",
        "latitude",
        "longitude"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).astype(
            "float64"
        )

    df["beneficiary_count"] = pd.to_numeric(
        df["beneficiary_count"],
        errors="coerce"
    ).astype(
        "float64"
    )

    # ========================================================
    # ANOMALY 1
    # EXTREME PROJECT COST
    # ========================================================

    print(
        "Injecting extreme cost anomalies..."
    )

    cost_indices = random.sample(
        range(TOTAL_PROJECTS),
        30
    )

    for idx in cost_indices:

        sanctioned = float(
            df.at[
                idx,
                "sanctioned_amount"
            ]
        )

        inflated = (
            sanctioned *
            random.uniform(
                2.5,
                5.0
            )
        )

        df.at[
            idx,
            "sanctioned_amount"
        ] = float(
            round(
                inflated,
                2
            )
        )

        df.at[
            idx,
            "released_amount"
        ] = float(
            round(
                inflated *
                random.uniform(
                    0.75,
                    0.95
                ),
                2
            )
        )

        df.at[
            idx,
            "expenditure"
        ] = float(
            round(
                inflated *
                random.uniform(
                    0.65,
                    0.90
                ),
                2
            )
        )

        df.at[
            idx,
            "_demo_injected_issue"
        ] = "EXTREME_COST"

    # ========================================================
    # ANOMALY 2
    # EXPENDITURE ABOVE SANCTIONED AMOUNT
    # ========================================================

    print(
        "Injecting expenditure anomalies..."
    )

    expenditure_indices = random.sample(
        range(TOTAL_PROJECTS),
        25
    )

    for idx in expenditure_indices:

        sanctioned = float(
            df.at[
                idx,
                "sanctioned_amount"
            ]
        )

        df.at[
            idx,
            "expenditure"
        ] = float(
            round(
                sanctioned *
                random.uniform(
                    1.05,
                    1.45
                ),
                2
            )
        )

        df.at[
            idx,
            "_demo_injected_issue"
        ] = "EXPENDITURE_OVER_SANCTION"

    # ========================================================
    # ANOMALY 3
    # UNDER-UTILIZATION
    # ========================================================

    print(
        "Injecting under-utilization anomalies..."
    )

    underuse_indices = random.sample(
        range(TOTAL_PROJECTS),
        30
    )

    for idx in underuse_indices:

        released = float(
            df.at[
                idx,
                "released_amount"
            ]
        )

        df.at[
            idx,
            "expenditure"
        ] = float(
            round(
                released *
                random.uniform(
                    0.03,
                    0.15
                ),
                2
            )
        )

        df.at[
            idx,
            "project_status"
        ] = "Ongoing"

        df.at[
            idx,
            "_demo_injected_issue"
        ] = "UNDER_UTILIZATION"

    # ========================================================
    # ANOMALY 4
    # MAJOR PROJECT DELAYS
    # ========================================================

    print(
        "Injecting project delay anomalies..."
    )

    delay_indices = random.sample(
        range(TOTAL_PROJECTS),
        35
    )

    for idx in delay_indices:

        start = datetime.strptime(

            df.at[
                idx,
                "start_date"
            ],

            "%Y-%m-%d"
        )

        delayed_completion = (

            start +

            timedelta(
                days=random.randint(
                    400,
                    900
                )
            )
        )

        df.at[
            idx,
            "completion_date"
        ] = delayed_completion.strftime(
            "%Y-%m-%d"
        )

        df.at[
            idx,
            "project_status"
        ] = "Delayed"

        df.at[
            idx,
            "_demo_injected_issue"
        ] = "MAJOR_DELAY"

    # ========================================================
    # ANOMALY 5
    # SIMILAR / DUPLICATE PROJECTS
    # ========================================================

    print(
        "Injecting duplicate/similar projects..."
    )

    duplicate_sources = random.sample(
        range(TOTAL_PROJECTS),
        25
    )

    for counter, source_idx in enumerate(
        duplicate_sources,
        start=1
    ):

        target_idx = (
            TOTAL_PROJECTS -
            counter
        )

        source = df.iloc[
            source_idx
        ].copy()

        source["project_id"] = (
            f"MPLAD-2026-DUP-{counter:04d}"
        )

        source["project_name"] = (

            str(
                source[
                    "project_name"
                ]
            )

            + " - Phase II"
        )

        source["sanctioned_amount"] = float(
            round(
                float(
                    source[
                        "sanctioned_amount"
                    ]
                )
                *
                random.uniform(
                    0.95,
                    1.05
                ),
                2
            )
        )

        source["released_amount"] = float(
            round(
                float(
                    source[
                        "released_amount"
                    ]
                )
                *
                random.uniform(
                    0.95,
                    1.05
                ),
                2
            )
        )

        source["expenditure"] = float(
            round(
                float(
                    source[
                        "expenditure"
                    ]
                )
                *
                random.uniform(
                    0.90,
                    1.10
                ),
                2
            )
        )

        source[
            "_demo_injected_issue"
        ] = "SIMILAR_DUPLICATE"

        df.iloc[
            target_idx
        ] = source

    # ========================================================
    # ANOMALY 6
    # CONTRACTOR CONCENTRATION
    # ========================================================

    print(
        "Injecting contractor concentration pattern..."
    )

    concentration_contractor = (
        "Unity Infrastructure"
    )

    concentration_indices = random.sample(
        range(TOTAL_PROJECTS),
        100
    )

    for idx in concentration_indices:

        df.at[
            idx,
            "contractor"
        ] = concentration_contractor

        current_issue = str(
            df.at[
                idx,
                "_demo_injected_issue"
            ]
        )

        if current_issue == "NORMAL":

            df.at[
                idx,
                "_demo_injected_issue"
            ] = (
                "CONTRACTOR_CONCENTRATION"
            )

    # ========================================================
    # ANOMALY 7
    # GEOGRAPHIC CLUSTER
    # ========================================================

    print(
        "Injecting geographic clustering..."
    )

    cluster_indices = random.sample(
        range(TOTAL_PROJECTS),
        40
    )

    cluster_lat = 17.3850

    cluster_lon = 78.4867

    for idx in cluster_indices:

        df.at[
            idx,
            "latitude"
        ] = float(
            round(
                cluster_lat +
                np.random.normal(
                    0,
                    0.002
                ),
                6
            )
        )

        df.at[
            idx,
            "longitude"
        ] = float(
            round(
                cluster_lon +
                np.random.normal(
                    0,
                    0.002
                ),
                6
            )
        )

        current_issue = str(
            df.at[
                idx,
                "_demo_injected_issue"
            ]
        )

        if current_issue == "NORMAL":

            df.at[
                idx,
                "_demo_injected_issue"
            ] = (
                "GEOGRAPHIC_CLUSTER"
            )

    # ========================================================
    # ANOMALY 8
    # MISSING DATA
    # ========================================================

    print(
        "Injecting data-quality problems..."
    )

    missing_indices = random.sample(
        range(TOTAL_PROJECTS),
        40
    )

    for idx in missing_indices:

        field = random.choice(

            [
                "contractor",
                "beneficiary_count",
                "completion_date",
                "latitude",
                "longitude",
                "released_amount"
            ]
        )

        df.at[
            idx,
            field
        ] = np.nan

        current_issue = str(
            df.at[
                idx,
                "_demo_injected_issue"
            ]
        )

        if current_issue == "NORMAL":

            df.at[
                idx,
                "_demo_injected_issue"
            ] = (
                "DATA_QUALITY"
            )

    # ========================================================
    # ANOMALY 9
    # INVALID NUMERICAL DATA
    # ========================================================

    print(
        "Injecting invalid numerical values..."
    )

    invalid_indices = random.sample(
        range(TOTAL_PROJECTS),
        15
    )

    for idx in invalid_indices:

        df.at[
            idx,
            "beneficiary_count"
        ] = float(
            -random.randint(
                1,
                500
            )
        )

        current_issue = str(
            df.at[
                idx,
                "_demo_injected_issue"
            ]
        )

        if current_issue == "NORMAL":

            df.at[
                idx,
                "_demo_injected_issue"
            ] = (
                "INVALID_NUMERICAL_DATA"
            )

    # ========================================================
    # ANOMALY 10
    # INVALID DATE
    # ========================================================

    print(
        "Injecting invalid date values..."
    )

    invalid_date_indices = random.sample(
        range(TOTAL_PROJECTS),
        10
    )

    for idx in invalid_date_indices:

        df.at[
            idx,
            "completion_date"
        ] = "INVALID_DATE"

        current_issue = str(
            df.at[
                idx,
                "_demo_injected_issue"
            ]
        )

        if current_issue == "NORMAL":

            df.at[
                idx,
                "_demo_injected_issue"
            ] = (
                "INVALID_DATE"
            )

    # ========================================================
    # FINAL TYPE CLEANUP
    # ========================================================

    # Keep financial fields as floats.
    for column in [

        "sanctioned_amount",
        "released_amount",
        "expenditure",
        "latitude",
        "longitude"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).astype(
            "float64"
        )

    # Beneficiary count can contain NaN
    # because we intentionally injected
    # missing values.
    df["beneficiary_count"] = pd.to_numeric(
        df["beneficiary_count"],
        errors="coerce"
    ).astype(
        "float64"
    )

    # ========================================================
    # SHUFFLE
    # ========================================================

    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(
        drop=True
    )

    # ========================================================
    # CREATE DATA DIRECTORY
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 70)

    print(
        "DATASET GENERATION COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"Total projects: {len(df)}"
    )

    print()

    print(
        f"Output file:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Dataset shape:"
    )

    print(
        df.shape
    )

    print()

    print(
        "Demo issue distribution:"
    )

    print(
        df[
            "_demo_injected_issue"
        ].value_counts()
    )

    print()

    print(
        "Columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The _demo_injected_issue column exists"
    )

    print(
        "ONLY for synthetic-data testing."
    )

    print(
        "The production AI pipeline will"
    )

    print(
        "NOT use this column."
    )

    print()

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    generate_dataset()