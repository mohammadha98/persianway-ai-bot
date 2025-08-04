from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Templates directory
TEMPLATES_DIR = os.path.join(BASE_DIR, "app", "templates")

# Static files directory
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")

# Initialize Jinja2Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

def configure_templates(app: FastAPI):
    """Configure Jinja2 templates and static files for the FastAPI app."""
    # Mount static files
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    
    return templates