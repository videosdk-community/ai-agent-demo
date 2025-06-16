import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings and configuration."""
    
    # Server Configuration
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "127.0.0.1")
    RELOAD: bool = os.getenv("RELOAD", "True").lower() == "true"
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    VIDEOSDK_AUTH_TOKEN: Optional[str] = os.getenv("VIDEOSDK_AUTH_TOKEN")
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    # Agent Configuration
    DEFAULT_TEMPERATURE: float = 0.8
    DEFAULT_TOP_P: float = 0.8
    DEFAULT_TOP_K: int = 40
    DEFAULT_VOICE: str = "Nova"
    
    # Session Configuration
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: int = 5000
    SESSION_TIMEOUT: int = 30
    
    def validate_required_keys(self) -> bool:
        """Validate that required API keys are present."""
        required_keys = [
            ("GOOGLE_API_KEY", self.GOOGLE_API_KEY),
            ("OPENAI_API_KEY", self.OPENAI_API_KEY),
        ]
        
        missing_keys = [key for key, value in required_keys if not value]
        
        if missing_keys:
            print(f"Warning: Missing required environment variables: {', '.join(missing_keys)}")
            return False
        return True


settings = Settings() 