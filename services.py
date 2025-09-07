import os
import json
import httpx
import PyPDF2
from io import BytesIO
from typing import Dict, Any, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("LLM_API_KEY"))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def summarize_report(self, text: str) -> Dict[str, Any]:
        """Summarize medical report and extract key findings"""
        prompt = f"""
        Analyze the following medical report and provide a structured summary in JSON format:
        
        {text}
        
        Return a JSON object with the following structure:
        {{
            "summary_text": "Brief overall summary of the report",
            "key_findings": ["finding1", "finding2", "finding3"],
            "recommendations": ["recommendation1", "recommendation2"],
            "abnormal_values": ["value1", "value2"],
            "follow_up_needed": true/false
        }}
        
        Focus on medical significance and patient-relevant information.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant that analyzes medical reports and provides structured summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_symptoms(self, message: str, user_history: Optional[str] = None) -> Dict[str, Any]:
        """Analyze symptoms and provide medical guidance"""
        context = f"\nUser's medical history context: {user_history}" if user_history else ""
        
        prompt = f"""
        Analyze the following symptom description and provide medical guidance in JSON format:
        
        Symptoms: {message}{context}
        
        Return a JSON object with the following structure:
        {{
            "possible_conditions": ["condition1", "condition2", "condition3"],
            "urgency": "low|medium|high|critical",
            "recommended_actions": ["action1", "action2", "action3"],
            "warning_signs": ["sign1", "sign2"],
            "when_to_seek_care": "immediate|within_24_hours|within_week|routine"
        }}
        
        Be conservative with urgency levels and always recommend professional medical consultation when appropriate.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant that provides preliminary symptom analysis. Always emphasize the importance of professional medical consultation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)

class PDFParserService:
    @staticmethod
    def parse_pdf(file_content: bytes) -> str:
        """Extract text from PDF file"""
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
