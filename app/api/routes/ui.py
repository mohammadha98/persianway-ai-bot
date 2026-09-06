from fastapi import APIRouter, Request, Depends, Form, HTTPException, File, UploadFile, Body
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import os
from typing import Optional, List, Dict, Any
import uuid
import httpx
import json
import datetime
from io import BytesIO

from app.core.templates import templates

# Create router for UI endpoints
router = APIRouter(prefix="/ui", tags=["UI"])

# Base URL for API calls
API_BASE_URL = "http://localhost:8000/api"

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page with tabs for chat, settings, and contribute."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "active_page": "home"}
    )

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Render the chat interface page."""
    # Generate a user ID if not present in session
    user_id = str(uuid.uuid4())
    
    # Get current time for initial message
    current_time = datetime.datetime.now().strftime("%H:%M")
    
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request, 
            "user_id": user_id, 
            "active_page": "chat",
            "current_time": current_time
        }
    )

@router.post("/chat/send")
async def send_message(request: Request, message: dict = Body(...)):
    """Process a chat message and return the response."""
    try:
        # Call the chat API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/chat/",
                json={
                    "message": message["message"],
                    "user_id": message.get("user_id", "default-user")
                }
            )
            
            if response.status_code != 200:
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": "Failed to get response from chat API"}
                )
            
            result = response.json()
            
            # Return the response
            return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Chat processing error: {str(e)}"}
        )

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Render the settings page."""
    try:
        # Get knowledge base status from API
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/knowledge-base/status")
            
            if response.status_code != 200:
                return templates.TemplateResponse(
                    "settings.html",
                    {
                        "request": request, 
                        "error": "Failed to get knowledge base status", 
                        "active_page": "settings",
                        "status": "unknown",
                        "document_counts": {"total": 0, "pdf": 0, "excel_qa": 0},
                        "rag_config": "Unknown",
                        "processing_status": "Unknown",
                        "last_updated": "Unknown",
                        "embedding_model": "Unknown",
                        "vector_store": "Unknown",
                        "chunk_size": "Unknown",
                        "chunk_overlap": "Unknown"
                    }
                )
            
            status_data = response.json()
            
            # Set default values for missing fields
            if "document_counts" not in status_data:
                status_data["document_counts"] = {"total": 0, "pdf": 0, "excel_qa": 0}
                
            return templates.TemplateResponse(
                "settings.html",
                {
                    "request": request, 
                    "status": status_data.get("status", "unknown"), 
                    "active_page": "settings",
                    "document_counts": status_data.get("document_counts", {"total": 0, "pdf": 0, "excel_qa": 0}),
                    "rag_config": "Persian Agriculture RAG System",
                    "processing_status": status_data.get("processing_status", "Idle"),
                    "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_model": "SentenceTransformers",
                    "vector_store": "FAISS",
                    "chunk_size": "1000",
                    "chunk_overlap": "200"
                }
            )
    except Exception as e:
        return templates.TemplateResponse(
            "settings.html",
            {
                "request": request, 
                "error": str(e), 
                "active_page": "settings",
                "status": "error",
                "document_counts": {"total": 0, "pdf": 0, "excel_qa": 0},
                "rag_config": "Unknown",
                "processing_status": "Error",
                "last_updated": "Unknown",
                "embedding_model": "Unknown",
                "vector_store": "Unknown",
                "chunk_size": "Unknown",
                "chunk_overlap": "Unknown"
            }
        )

@router.post("/settings/process-pdf")
async def process_pdf(request: Request, file: UploadFile = File(...), description: str = Form("")):
    """Process PDF document and return status."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Create form data
        form = aiohttp.FormData()
        form.add_field('file', file_content, filename=file.filename, content_type=file.content_type)
        form.add_field('description', description)
        
        # Call the knowledge base API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/knowledge-base/process-pdf",
                files={"file": (file.filename, file_content, file.content_type)},
                data={"description": description}
            )
            
            if response.status_code != 200:
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": "Failed to process PDF document"}
                )
            
            result = response.json()
            
            # Return the response
            return JSONResponse(content={"success": True, "message": "PDF document submitted for processing"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Document processing error: {str(e)}"}
        )

@router.post("/settings/process-excel")
async def process_excel(request: Request, file: UploadFile = File(...), description: str = Form("")):
    """Process Excel QA file and return status."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Call the knowledge base API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/knowledge-base/process-excel",
                files={"file": (file.filename, file_content, file.content_type)},
                data={"description": description}
            )
            
            if response.status_code != 200:
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": "Failed to process Excel file"}
                )
            
            result = response.json()
            
            # Return the response
            return JSONResponse(content={"success": True, "message": "Excel file submitted for processing"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Excel processing error: {str(e)}"}
        )


@router.get("/contribute", response_class=HTMLResponse)
async def contribute_page(request: Request):
    """Render the knowledge contribution page."""
    return templates.TemplateResponse(
        "contribute.html",
        {"request": request, "active_page": "contribute"}
    )

@router.post("/contribute/submit")
async def submit_contribution(request: Request, 
                             title: str = Form(...),
                             content: str = Form(...),
                             source_type: str = Form(...),
                             source_name: str = Form(...),
                             author: str = Form(""),
                             publication_date: str = Form(""),
                             tags: str = Form(""),
                             notes: str = Form("")):
    """Submit a knowledge contribution."""
    try:
        # Prepare the contribution data
        contribution_data = {
            "title": title,
            "content": content,
            "source_type": source_type,
            "source_name": source_name,
            "author": author,
            "publication_date": publication_date,
            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
            "notes": notes,
            "submission_date": datetime.datetime.now().isoformat()
        }
        
        # Call the knowledge base API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/knowledge-base/contribute",
                json=contribution_data
            )
            
            if response.status_code != 200:
                return JSONResponse(
                    status_code=response.status_code,
                    content={"error": "Failed to submit contribution"}
                )
            
            result = response.json()
            
            # Return the response
            return JSONResponse(content={"success": True, "message": "Contribution submitted successfully"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Contribution submission error: {str(e)}"}
        )