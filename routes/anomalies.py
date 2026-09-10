from flask import Blueprint, jsonify

from ml.pipeline import (
    run_anomaly_pipeline,
    get_anomaly_summary
)


# ============================================================
# MPLAD ANOMALY API
# ============================================================
#
# Provides Flask endpoints for:
#
#   POST /api/anomalies/run
#       Run Isolation Forest on all database projects.
#
#   GET /api/anomalies
#       Return the current anomaly summary.
#
# ============================================================


anomalies_bp = Blueprint(
    "anomalies",
    __name__,
    url_prefix="/api/anomalies"
)


# ------------------------------------------------------------
# Run AI anomaly detection
# ------------------------------------------------------------

@anomalies_bp.route("/run", methods=["POST"])
def run_anomalies():

    try:

        result = run_anomaly_pipeline()

        if not result.get("success", False):

            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as error:

        return jsonify({
            "success": False,
            "message": "AI anomaly detection failed.",
            "error": str(error)
        }), 500


# ------------------------------------------------------------
# Get anomaly summary
# ------------------------------------------------------------

@anomalies_bp.route("", methods=["GET"])
def anomaly_summary():

    try:

        summary = get_anomaly_summary()

        return jsonify({
            "success": True,
            **summary
        }), 200

    except Exception as error:

        return jsonify({
            "success": False,
            "message": "Unable to retrieve anomaly summary.",
            "error": str(error)
        }), 500


# ------------------------------------------------------------
# Health check for anomaly service
# ------------------------------------------------------------

@anomalies_bp.route("/health", methods=["GET"])
def anomaly_health():

    return jsonify({
        "success": True,
        "service": "MPLAD Isolation Forest Anomaly Detection",
        "status": "ready"
    }), 200
