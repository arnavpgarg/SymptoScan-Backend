from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MessageType(str, Enum):
    USER = "user"
    AI = "ai"

# Request Models
class SummarizeReportRequest(BaseModel):
    document_id: Optional[str] = None
    parsed_text: Optional[str] = None

class SymptomChatRequest(BaseModel):
    user_id: str
    message: str

class TTSRequest(BaseModel):
    text: str

# Response Models
class DocumentResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    file_path: str
    upload_date: datetime
    file_size: int

class SummaryResponse(BaseModel):
    id: str
    document_id: str
    summary_text: str
    key_findings: List[str]
    recommendations: List[str]
    created_at: datetime

class ChatResponse(BaseModel):
    possible_conditions: List[str]
    urgency: UrgencyLevel
    recommended_actions: List[str]

class MessageResponse(BaseModel):
    id: str
    user_id: str
    message_type: MessageType
    content: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

class HistoryResponse(BaseModel):
    documents: List[DocumentResponse]
    summaries: List[SummaryResponse]
    messages: List[MessageResponse]

class UploadResponse(BaseModel):
    document_id: str
    message: str
    file_path: str

class TTSResponse(BaseModel):
    audio_url: Optional[str] = None
    audio_data: Optional[bytes] = None
    message: str
