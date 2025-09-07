import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseClient:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        # Check if we're using mock credentials for development
        if self.url.startswith("https://mock-") or self.key.startswith("mock_"):
            self.client = None  # Mock client for development
        else:
            self.client: Client = create_client(self.url, self.key)
    
    def get_client(self) -> Client:
        if self.client is None:
            raise ValueError("Mock Supabase client - API calls not available in development mode")
        return self.client

# Global instance
supabase_client = SupabaseClient()
