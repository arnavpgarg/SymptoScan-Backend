# SymptoScan Backend API

A FastAPI backend for the SymptoScan medical report analysis and symptom chat application.

## Features

- **Medical Report Upload**: Upload PDF, image, and text files to Supabase Storage
- **AI-Powered Report Summarization**: Extract key findings and recommendations using GPT-4
- **Symptom Analysis Chat**: Interactive symptom analysis with medical guidance
- **Text-to-Speech**: Convert medical summaries to audio using ElevenLabs
- **User History**: Complete history tracking for documents, summaries, and conversations
- **Retry Logic**: Robust error handling with automatic retries for external API calls

## API Endpoints

### 1. POST `/upload-report`
Upload a medical report file and create a document record.

**Request**: Multipart form data
- `user_id`: String (form field)
- `file`: File upload (PDF, image, or text)

**Response**: Document ID and file path

### 2. POST `/summarize-report`
Analyze and summarize a medical report using AI.

**Request Body**:
```json
{
  "document_id": "uuid", // OR
  "parsed_text": "raw text content"
}
```

**Response**: Structured summary with key findings and recommendations

### 3. POST `/symptom-chat`
Interactive symptom analysis with medical guidance.

**Request Body**:
```json
{
  "user_id": "string",
  "message": "symptom description"
}
```

**Response**: Possible conditions, urgency level, and recommended actions

### 4. GET `/history/{user_id}`
Retrieve complete user history including documents, summaries, and messages.

**Response**: Complete user history data

### 5. POST `/tts`
Convert text to speech using ElevenLabs API.

**Request Body**:
```json
{
  "text": "text to convert"
}
```

**Response**: Audio file (MP3 format)

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required environment variables:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase anon key
- `LLM_API_KEY`: Your OpenAI API key
- `ELEVENLABS_KEY`: Your ElevenLabs API key

### 3. Supabase Database Schema

Create the following tables in your Supabase database:

#### Documents Table
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    file_size INTEGER NOT NULL
);
```

#### Summaries Table
```sql
CREATE TABLE summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    summary_text TEXT NOT NULL,
    key_findings TEXT[] DEFAULT '{}',
    recommendations TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Messages Table
```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    message_type TEXT NOT NULL CHECK (message_type IN ('user', 'ai')),
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 4. Supabase Storage
Create a storage bucket named `reports` in your Supabase project for file uploads.

### 5. Run the Server
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Documentation

Once the server is running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## CORS Configuration

The API is configured to accept requests from:
- Lovable.dev domains
- Local development servers (localhost:3000, localhost:5173)

## Error Handling

- All external API calls include retry logic with exponential backoff
- Comprehensive error messages for debugging
- Proper HTTP status codes for different error scenarios

## Security Considerations

- Environment variables for sensitive API keys
- File type validation for uploads
- Input validation using Pydantic models
- CORS restrictions for security

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Supabase**: Backend-as-a-Service for database and storage
- **OpenAI**: GPT-4 for medical text analysis
- **ElevenLabs**: Text-to-speech conversion
- **PyPDF2**: PDF text extraction
- **Tenacity**: Retry logic for external API calls
