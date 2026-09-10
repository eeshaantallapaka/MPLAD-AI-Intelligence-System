from flask import Blueprint, jsonify, request, render_template

from database.models import Project


projects_bp = Blueprint(
    "projects",
    __name__
)


# ============================================================
# PROJECT REGISTRY
# ============================================================

@projects_bp.route(
    "/api/projects",
    methods=["GET"]
)
def get_projects():

    query = Project.query

    project_id = request.args.get("project_id")
    state = request.args.get("state")
    district = request.args.get("district")
    constituency = request.args.get("constituency")
    status = request.args.get("status")
    risk_level = request.args.get("risk_level")
    category = request.args.get("category")
    contractor = request.args.get("contractor")

    if project_id:
        query = query.filter(
            Project.project_id.ilike(
                f"%{project_id}%"
            )
        )

    if state:
        query = query.filter(
            Project.state.ilike(
                f"%{state}%"
            )
        )

    if district:
        query = query.filter(
            Project.district.ilike(
                f"%{district}%"
            )
        )

    if constituency:
        query = query.filter(
            Project.constituency.ilike(
                f"%{constituency}%"
            )
        )

    if status:
        query = query.filter(
            Project.project_status.ilike(
                f"%{status}%"
            )
        )

    if risk_level:
        query = query.filter(
            Project.risk_level ==
            risk_level.upper()
        )

    if category:
        query = query.filter(
            Project.project_category.ilike(
                f"%{category}%"
            )
        )

    if contractor:
        query = query.filter(
            Project.contractor.ilike(
                f"%{contractor}%"
            )
        )

    projects = query.order_by(
        Project.risk_score.desc()
    ).all()

    return jsonify({
        "success": True,
        "count": len(projects),
        "data": [
            project.to_dict()
            for project in projects
        ]
    })


# ============================================================
# SINGLE PROJECT API
# ============================================================

@projects_bp.route(
    "/api/projects/<project_id>",
    methods=["GET"]
)
def get_project(project_id):

    project = Project.query.filter_by(
        project_id=project_id
    ).first()

    if not project:

        return jsonify({
            "success": False,
            "error": "Project not found."
        }), 404

    return jsonify({
        "success": True,
        "data": project.to_dict()
    })


# ============================================================
# PROJECT DETAILS PAGE
# ============================================================

@projects_bp.route(
    "/projects/<project_id>",
    methods=["GET"]
)
def project_details(project_id):

    project = Project.query.filter_by(
        project_id=project_id
    ).first()

    if not project:
        return (
            "Project not found.",
            404
        )

    return render_template(
        "project_detail.html",
        project_id=project_id
    )