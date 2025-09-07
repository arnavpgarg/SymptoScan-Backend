from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uuid
from datetime import datetime
from typing import Optional
import asyncio

from models import (
    SummarizeReportRequest, SymptomChatRequest, TTSRequest,
    DocumentResponse, SummaryResponse, ChatResponse, MessageResponse,
    HistoryResponse, UploadResponse, TTSResponse, MessageType
)
from database import supabase_client
from services import llm_service, pdf_parser, tts_service

app = FastAPI(
    title="SymptoScan API",
    description="Backend API for SymptoScan medical report analysis and symptom chat",
    version="1.0.0"
)

# CORS configuration for Lovable and other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lovable.dev",
        "https://*.lovable.dev",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "SymptoScan API is running", "version": "1.0.0"}

@app.post("/upload-report", response_model=UploadResponse)
async def upload_report(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload a medical report file to Supabase Storage and create a document record"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith(('application/pdf', 'image/', 'text/')):
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF, image, and text files are allowed.")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Generate unique filename
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'pdf'
        unique_filename = f"{user_id}/{uuid.uuid4()}.{file_extension}"
        
        # Upload to Supabase Storage
        supabase = supabase_client.get_client()
        storage_response = supabase.storage.from_("reports").upload(
            path=unique_filename,
            file=file_content,
            file_options={"content-type": file.content_type}
        )
        
        if hasattr(storage_response, 'error') and storage_response.error:
            raise HTTPException(status_code=500, detail=f"Failed to upload file: {storage_response.error}")
        
        # Create document record in database
        document_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "filename": file.filename,
            "storage_url": f"reports/{unique_filename}",
            "upload_date": datetime.utcnow().isoformat(),
            "file_size": file_size
        }
        
        db_response = supabase.table("documents").insert(document_data).execute()
        
        if hasattr(db_response, 'error') and db_response.error:
            raise HTTPException(status_code=500, detail=f"Failed to create document record: {db_response.error}")
        
        return UploadResponse(
            document_id=document_data["id"],
            message="File uploaded successfully",
            file_path=unique_filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/summarize-report", response_model=SummaryResponse)
async def summarize_report(request: SummarizeReportRequest):
    """Summarize a medical report using LLM"""
    try:
        text_to_analyze = ""
        document_id = None
        
        if request.document_id:
            # Get document from database
            supabase = supabase_client.get_client()
            doc_response = supabase.table("documents").select("*").eq("id", request.document_id).execute()
            
            if not doc_response.data:
                raise HTTPException(status_code=404, detail="Document not found")
            
            document = doc_response.data[0]
            document_id = document["id"]
            
            # Download file from Supabase Storage
            file_response = supabase.storage.from_("reports").download(document["storage_url"])
            
            if hasattr(file_response, 'error') and file_response.error:
                raise HTTPException(status_code=500, detail=f"Failed to download file: {file_response.error}")
            
            # Parse PDF if needed
            if document["filename"].lower().endswith('.pdf'):
                text_to_analyze = pdf_parser.parse_pdf(file_response)
            else:
                text_to_analyze = file_response.decode('utf-8')
                
        elif request.parsed_text:
            text_to_analyze = request.parsed_text
            document_id = str(uuid.uuid4())  # Generate ID for standalone text analysis
        else:
            raise HTTPException(status_code=400, detail="Either document_id or parsed_text must be provided")
        
        # Call LLM for summarization
        summary_data = await llm_service.summarize_report(text_to_analyze)
        
        # Save summary to database
        summary_record = {
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "summary_text": summary_data.get("summary_text", ""),
            "key_findings": summary_data.get("key_findings", []),
            "recommendations": summary_data.get("recommendations", []),
            "created_at": datetime.utcnow().isoformat()
        }
        
        supabase = supabase_client.get_client()
        db_response = supabase.table("summaries").insert(summary_record).execute()
        
        if hasattr(db_response, 'error') and db_response.error:
            raise HTTPException(status_code=500, detail=f"Failed to save summary: {db_response.error}")
        
        return SummaryResponse(**summary_record)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/symptom-chat", response_model=ChatResponse)
async def symptom_chat(request: SymptomChatRequest):
    """Analyze symptoms and provide medical guidance"""
    try:
        # Save user message
        user_message = {
            "id": str(uuid.uuid4()),
            "user_id": request.user_id,
            "message_type": MessageType.USER,
            "content": request.message,
            "created_at": datetime.utcnow().isoformat()
        }
        
        supabase = supabase_client.get_client()
        supabase.table("messages").insert(user_message).execute()
        
        # Get user's recent medical history for context
        history_response = supabase.table("messages").select("content").eq("user_id", request.user_id).order("created_at", desc=True).limit(10).execute()
        
        user_history = ""
        if history_response.data:
            user_history = " ".join([msg["content"] for msg in history_response.data])
        
        # Analyze symptoms with LLM
        analysis = await llm_service.analyze_symptoms(request.message, user_history)
        
        # Save AI response
        ai_message = {
            "id": str(uuid.uuid4()),
            "user_id": request.user_id,
            "message_type": MessageType.AI,
            "content": f"Analysis: {analysis}",
            "metadata": analysis,
            "created_at": datetime.utcnow().isoformat()
        }
        
        supabase.table("messages").insert(ai_message).execute()
        
        return ChatResponse(
            possible_conditions=analysis.get("possible_conditions", []),
            urgency=analysis.get("urgency", "low"),
            recommended_actions=analysis.get("recommended_actions", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symptom analysis failed: {str(e)}")

@app.get("/history/{user_id}", response_model=HistoryResponse)
async def get_user_history(user_id: str):
    """Get user's complete history including documents, summaries, and messages"""
    try:
        supabase = supabase_client.get_client()
        
        # Get documents
        docs_response = supabase.table("documents").select("*").eq("user_id", user_id).order("upload_date", desc=True).execute()
        documents = [DocumentResponse(**doc) for doc in docs_response.data] if docs_response.data else []
        
        # Get summaries for user's documents
        doc_ids = [doc.id for doc in documents]
        summaries = []
        if doc_ids:
            summaries_response = supabase.table("summaries").select("*").in_("document_id", doc_ids).order("created_at", desc=True).execute()
            summaries = [SummaryResponse(**summary) for summary in summaries_response.data] if summaries_response.data else []
        
        # Get messages
        messages_response = supabase.table("messages").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        messages = [MessageResponse(**msg) for msg in messages_response.data] if messages_response.data else []
        
        return HistoryResponse(
            documents=documents,
            summaries=summaries,
            messages=messages
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using ElevenLabs API"""
    try:
        # Generate audio
        audio_data = await tts_service.text_to_speech(request.text)
        
        # Return audio data directly
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
