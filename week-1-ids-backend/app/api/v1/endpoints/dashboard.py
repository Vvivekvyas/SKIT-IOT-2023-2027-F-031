"""
HTML page routes (not JSON) — these render the Jinja2 templates for the
'simple, clean dashboard UI' described in the proposal's Designs slide.
The dashboard's JS calls the JSON endpoints in traffic.py / predict.py
to populate the table and alert sections.
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/dashboard")
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})
