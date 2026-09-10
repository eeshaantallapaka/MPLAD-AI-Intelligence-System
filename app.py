from flask import Flask, jsonify, render_template

from config import Config

from database.db import db

from routes.dashboard import dashboard_bp
from routes.projects import projects_bp
from routes.upload import upload_bp
from routes.anomalies import anomalies_bp
from routes.reports import reports_bp


def create_app():

    app = Flask(__name__)

    app.config.from_object(Config)

    # =========================================================
    # DATABASE
    # =========================================================

    db.init_app(app)

    # =========================================================
    # REGISTER BLUEPRINTS
    # =========================================================

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(anomalies_bp)
    app.register_blueprint(reports_bp)

    # =========================================================
    # CREATE DATABASE TABLES
    # =========================================================

    with app.app_context():
        db.create_all()

    # =========================================================
    # DASHBOARD
    # =========================================================

    @app.route("/")
    def index():

        return render_template(
            "dashboard.html"
        )

    # =========================================================
    # PROJECTS PAGE
    # =========================================================

    @app.route("/projects")
    def projects_page():

        return render_template(
            "projects.html"
        )

    # =========================================================
    # ANOMALIES PAGE
    # =========================================================

    @app.route("/anomalies")
    def anomalies_page():

        return render_template(
            "anomalies.html"
        )

    # =========================================================
    # UPLOAD DATA PAGE
    # =========================================================

    @app.route("/upload")
    def upload_page():

        return render_template(
            "upload.html"
        )

    # =========================================================
    # REPORTS PAGE
    # =========================================================

    @app.route("/reports")
    def reports_page():

        return render_template(
            "reports.html"
        )

    # =========================================================
    # HEALTH CHECK
    # =========================================================

    @app.route("/health")
    def health():

        return jsonify({
            "success": True,
            "status": "healthy",
            "application": "MPLAD AI Intelligence System"
        })

    # =========================================================
    # 404 ERROR
    # =========================================================

    @app.errorhandler(404)
    def not_found(error):

        return jsonify({
            "success": False,
            "error": "Resource not found."
        }), 404

    # =========================================================
    # FILE TOO LARGE
    # =========================================================

    @app.errorhandler(413)
    def file_too_large(error):

        return jsonify({
            "success": False,
            "error": "Uploaded file is too large."
        }), 413

    # =========================================================
    # INTERNAL SERVER ERROR
    # =========================================================

    @app.errorhandler(500)
    def internal_error(error):

        return jsonify({
            "success": False,
            "error": "Internal server error."
        }), 500

    return app


# =============================================================
# CREATE APPLICATION
# =============================================================

app = create_app()


# =============================================================
# RUN APPLICATION
# =============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )