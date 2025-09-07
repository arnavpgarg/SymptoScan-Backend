import os
import json
import httpx
import PyPDF2
from io import BytesIO
from typing import Dict, Any, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from dotenv import load_dotenv
from document_parser import advanced_parser

load_dotenv()

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("LLM_API_KEY"))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def summarize_report(self, text: str) -> Dict[str, Any]:
        """Summarize medical report and extract key findings"""
        prompt = f"""SYSTEM: You are a precise medical assistant. Output only JSON inside triple backticks labeled json, with fields:

{{
  "patient_name": "",
  "age": "",
  "gender": "",
  "lab_results": {{ "Test Name": "value units", ...}},
  "summary_text": ""  // 2-4 sentence explanation in layman terms, one or two action items
}}

USER: Here is the report content: {text}

Notes: If a specific field is missing, set it to null. Keep tone empathetic. At the end also output a plain-language one-line urgency tag (low/medium/high) outside the JSON."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant that analyzes medical reports and provides structured summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # Parse response that contains JSON in triple backticks plus urgency tag
        content = response.choices[0].message.content
        
        # Extract JSON from triple backticks
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            parsed_data = json.loads(json_str)
        else:
            # Fallback: try to parse entire content as JSON
            parsed_data = json.loads(content)
        
        # Extract urgency tag (outside JSON)
        urgency_match = re.search(r'(?:urgency|priority):\s*(low|medium|high)', content.lower())
        if urgency_match:
            parsed_data["urgency"] = urgency_match.group(1)
        else:
            # Look for standalone urgency words
            urgency_match = re.search(r'\b(low|medium|high)\s*(?:urgency|priority)?\b', content.lower())
            if urgency_match:
                parsed_data["urgency"] = urgency_match.group(1)
            else:
                parsed_data["urgency"] = "medium"  # Default fallback
        
        return parsed_data
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_symptoms(self, message: str, user_history: Optional[str] = None) -> Dict[str, Any]:
        """Analyze symptoms and provide medical guidance"""
        context = f"\nUser's medical history context: {user_history}" if user_history else ""
        
        prompt = f"""SYSTEM: You are a triage assistant (informational only). Output JSON only inside triple backticks labeled json:
{{
  "possible_conditions": ["..."],
  "urgency": "low|medium|high",
  "recommended_actions": "short layman-friendly steps"
}}
USER: Patient reports: "{message}". Use conservative judgment: when in doubt, mark urgency = high."""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant that provides preliminary symptom analysis. Always emphasize the importance of professional medical consultation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Parse response that contains JSON in triple backticks
        content = response.choices[0].message.content
        
        # Extract JSON from triple backticks
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        else:
            # Fallback: try to parse entire content as JSON
            return json.loads(content)

class PDFParserService:
    @staticmethod
    async def parse_pdf(file_content: bytes, filename: str = "document.pdf", file_url: Optional[str] = None) -> str:
        """
        Advanced PDF parsing using LlamaParse/Unstructured APIs with OCR fallback
        
        Args:
            file_content: Raw file bytes
            filename: Original filename for API calls
            file_url: Optional signed URL for the file (used by Unstructured API)
        
        Returns:
            Extracted and structured text content
        """
        try:
            # Use advanced document parser with multiple parsing strategies
            return await advanced_parser.parse_document(file_content, filename, file_url)
        except Exception as e:
            # Final fallback to basic PyPDF2 extraction
            try:
                pdf_file = BytesIO(file_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip() if text.strip() else "No text could be extracted from this document."
            except Exception as fallback_error:
                raise ValueError(f"All parsing methods failed. Advanced parser: {str(e)}, Basic parser: {str(fallback_error)}")
    
    @staticmethod
    def parse_pdf_sync(file_content: bytes) -> str:
        """Synchronous fallback method for compatibility"""
        try:
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")

class ElevenLabsService:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def text_to_speech(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> bytes:
        """Convert text to speech using ElevenLabs API"""
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.content

# Service instances
llm_service = LLMService()
pdf_parser = PDFParserService()
tts_service = ElevenLabsService()
