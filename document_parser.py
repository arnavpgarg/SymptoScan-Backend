import os
import io
import base64
import asyncio
from typing import Dict, Any, Optional, Tuple
from io import BytesIO
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

class AdvancedDocumentParser:
    def __init__(self):
        self.llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")
        self.unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
        self.unstructured_url = os.getenv("UNSTRUCTURED_API_URL", "https://api.unstructured.io")
        
    async def parse_document(self, file_content: bytes, filename: str, file_url: Optional[str] = None) -> str:
        """
        Parse document using advanced parsing methods with fallback strategy:
        1. Try LlamaParse (best for complex layouts)
        2. Try Unstructured API (good for structured documents)
        3. Try OCR if document appears to be scanned
        4. Fallback to basic PyMuPDF extraction
        """
        try:
            # First, check if document is scanned/image-based
            is_scanned = await self._is_scanned_document(file_content)
            
            if is_scanned:
                print("Document appears to be scanned, using OCR...")
                return await self._extract_with_ocr(file_content)
            
            # Try LlamaParse first (best for complex medical documents)
            if self.llama_parse_api_key:
                try:
                    result = await self._parse_with_llama_parse(file_content, filename)
                    if result:
                        return result
                except Exception as e:
                    print(f"LlamaParse failed: {e}")
            
            # Try Unstructured API
            if self.unstructured_api_key:
                try:
                    result = await self._parse_with_unstructured(file_content, filename, file_url)
                    if result:
                        return result
                except Exception as e:
                    print(f"Unstructured API failed: {e}")
            
            # Fallback to PyMuPDF with enhanced extraction
            return await self._extract_with_pymupdf(file_content)
            
        except Exception as e:
            print(f"All parsing methods failed: {e}")
            # Final fallback to basic text extraction
            return await self._basic_text_extraction(file_content)
    
    async def _is_scanned_document(self, file_content: bytes) -> bool:
        """Check if PDF is primarily image-based (scanned)"""
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            total_chars = 0
            total_images = 0
            
            for page_num in range(min(3, len(doc))):  # Check first 3 pages
                page = doc[page_num]
                text = page.get_text()
                images = page.get_images()
                
                total_chars += len(text.strip())
                total_images += len(images)
            
            doc.close()
            
            # If very little text but many images, likely scanned
            return total_chars < 100 and total_images > 0
            
        except Exception:
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _parse_with_llama_parse(self, file_content: bytes, filename: str) -> Optional[str]:
        """Parse document using LlamaParse API"""
        url = "https://api.cloud.llamaindex.ai/api/parsing/upload"
        
        headers = {
            "Authorization": f"Bearer {self.llama_parse_api_key}",
        }
        
        files = {
            "file": (filename, file_content, "application/pdf")
        }
        
        data = {
            "parsing_instruction": "Extract all text content from this medical document, preserving structure and formatting. Include headers, sections, tables, and any medical data.",
            "result_type": "text"
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Upload document
            response = await client.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            
            job_data = response.json()
            job_id = job_data.get("id")
            
            if not job_id:
                return None
            
            # Poll for results
            result_url = f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/text"
            
            for _ in range(30):  # Wait up to 5 minutes
                await asyncio.sleep(10)
                
                result_response = await client.get(result_url, headers=headers)
                
                if result_response.status_code == 200:
                    result_data = result_response.json()
                    if result_data.get("status") == "SUCCESS":
                        return result_data.get("text", "")
                elif result_response.status_code != 202:  # Not still processing
                    break
            
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _parse_with_unstructured(self, file_content: bytes, filename: str, file_url: Optional[str] = None) -> Optional[str]:
        """Parse document using Unstructured API"""
        url = f"{self.unstructured_url}/general/v0/general"
        
        headers = {
            "unstructured-api-key": self.unstructured_api_key,
        }
        
        # Use file URL if available, otherwise send file content
        if file_url:
            data = {
                "url": file_url,
                "strategy": "hi_res",
                "pdf_infer_table_structure": "true",
                "extract_image_block_types": '["Image", "Table"]'
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, data=data)
        else:
            files = {
                "files": (filename, file_content, "application/pdf")
            }
            
            data = {
                "strategy": "hi_res",
                "pdf_infer_table_structure": "true",
                "extract_image_block_types": '["Image", "Table"]'
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, files=files, data=data)
        
        response.raise_for_status()
        elements = response.json()
        
        # Combine all text elements
        text_parts = []
        for element in elements:
            if element.get("type") in ["Title", "NarrativeText", "ListItem", "Table"]:
                text = element.get("text", "").strip()
                if text:
                    text_parts.append(text)
        
        return "\n\n".join(text_parts) if text_parts else None
    
    async def _extract_with_ocr(self, file_content: bytes) -> str:
        """Extract text using OCR for scanned documents"""
        try:
            # Convert PDF to images
            images = convert_from_bytes(file_content, dpi=300)
            
            extracted_text = []
            
            for i, image in enumerate(images):
                # Use Tesseract OCR
                text = pytesseract.image_to_string(image, config='--psm 6')
                if text.strip():
                    extracted_text.append(f"--- Page {i+1} ---\n{text.strip()}")
            
            return "\n\n".join(extracted_text)
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return await self._extract_with_pymupdf(file_content)
    
    async def _extract_with_pymupdf(self, file_content: bytes) -> str:
        """Enhanced text extraction using PyMuPDF"""
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with layout preservation
                text = page.get_text("text")
                
                # Also try to extract tables
                tables = page.find_tables()
                for table in tables:
                    try:
                        table_data = table.extract()
                        if table_data:
                            table_text = "\n".join(["\t".join(row) for row in table_data if row])
                            text += f"\n\nTable:\n{table_text}\n"
                    except:
                        pass
                
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text.strip()}")
            
            doc.close()
            return "\n\n".join(text_parts)
            
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
            return await self._basic_text_extraction(file_content)
    
    async def _basic_text_extraction(self, file_content: bytes) -> str:
        """Basic fallback text extraction"""
        try:
            import PyPDF2
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            return f"Failed to extract text from document: {str(e)}"
    
    async def get_signed_url_from_supabase(self, supabase_client, file_path: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a signed URL for the file in Supabase Storage"""
        try:
            response = supabase_client.storage.from_("reports").create_signed_url(file_path, expires_in)
            return response.get("signedURL")
        except Exception as e:
            print(f"Failed to create signed URL: {e}")
            return None

# Global instance
advanced_parser = AdvancedDocumentParser()
