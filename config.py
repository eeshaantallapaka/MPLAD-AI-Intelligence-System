import os


BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)


class Config:

    # ========================================================
    # Flask
    # ========================================================

    SECRET_KEY = os.environ.get(
        "SECRET_KEY",
        "mplad-ai-development-secret-key"
    )

    # ========================================================
    # Database
    # ========================================================

    INSTANCE_DIR = os.path.join(
        BASE_DIR,
        "instance"
    )

    os.makedirs(
        INSTANCE_DIR,
        exist_ok=True
    )

    DATABASE_PATH = os.path.join(
        INSTANCE_DIR,
        "mplad.db"
    )

    SQLALCHEMY_DATABASE_URI = (
        "sqlite:///"
        + DATABASE_PATH
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ========================================================
    # Upload configuration
    # ========================================================

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    ALLOWED_EXTENSIONS = {
        ".csv",
        ".xlsx",
        ".xls"
    }

    # ========================================================
    # Risk weights
    # ========================================================

    RISK_WEIGHTS = {
        "financial": 0.30,
        "delay": 0.20,
        "duplicate": 0.20,
        "contractor": 0.15,
        "geographic": 0.10,
        "data_quality": 0.05
    }

    # ========================================================
    # Risk thresholds
    # ========================================================

    RISK_THRESHOLDS = {
        "LOW": 25,
        "MEDIUM": 50,
        "HIGH": 75,
        "CRITICAL": 100
    }

    # ========================================================
    # Isolation Forest
    # ========================================================

    ML_N_ESTIMATORS = 200
    ML_RANDOM_STATE = 42
    ML_CONTAMINATION = "auto"

    # Names expected by anomaly_detector.py
    ISOLATION_FOREST_ESTIMATORS = 200
    ISOLATION_FOREST_RANDOM_STATE = 42
    ISOLATION_FOREST_CONTAMINATION = "auto"

    # ========================================================
    # Duplicate detection
    # ========================================================

    DUPLICATE_SIMILARITY_THRESHOLD = 0.85
    DUPLICATE_SIMILARITY = 0.85

    # ========================================================
    # Financial analysis
    # ========================================================

    COST_MULTIPLIER_THRESHOLD = 2.0

    EXPENDITURE_TOLERANCE = 1.0

    # ========================================================
    # Contractor analysis
    # ========================================================

    CONTRACTOR_CONCENTRATION_THRESHOLD = 0.30

    # ========================================================
    # Delay analysis
    # ========================================================

    DEFAULT_EXPECTED_DURATION_DAYS = 180
    DEFAULT_EXPECTED_DURATION = 180

    # ========================================================
    # Application
    # ========================================================

    JSON_SORT_KEYS = False