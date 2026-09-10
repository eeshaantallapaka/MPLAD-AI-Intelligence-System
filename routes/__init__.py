from routes.reports import reports_bp
from routes.dashboard import dashboard_bp
from routes.projects import projects_bp
from routes.upload import upload_bp
from routes.anomalies import anomalies_bp


__all__ = [
    "dashboard_bp",
    "projects_bp",
    "upload_bp",
    "anomalies_bp"
]