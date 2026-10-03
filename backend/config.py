"""
MedRittAI — Application Configuration
Loads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Application ---
    APP_NAME: str = "MedRittAI"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "info"

    # --- Security ---
    SECRET_KEY: str = Field(..., description="Secret key required for session/JWT encryption")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 8

    # --- Database ---
    DATA_DIR: str = Field(default="./data", description="Root directory for runtime data")
    DATABASE_URL: str = Field(
        default="sqlite:///./data/medritt.db",
        description="SQLite connection string. Required.",
    )

    # --- Pre-inference scan type verification ---
    STRICT_SCAN_TYPE_VALIDATION: bool = True
    SCAN_TYPE_VERIFIER_MODEL: Optional[str] = None
    SCAN_TYPE_GROQ_MODEL: str = "qwen/qwen3.8-27b"
    SCAN_TYPE_MIN_CONFIDENCE: float = 0.85

    # --- Clinical report generation ---
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Gemini API Key for generation tasks"
    )
    GEMINI_MODEL: str = Field(
        default="gemini-1.5-flash",
        description="Preferred Gemini model for image-aware reports"
    )
    SARVAM_API_KEY: Optional[str] = Field(
        default=None,
        description="Sarvam API key used for patient-language translation"
    )
    SARVAM_TRANSLATE_MODEL: str = Field(
        default="sarvam-translate:v1",
        description="Sarvam text translation model"
    )
    GROQ_ENABLED: bool = True
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="Groq API key for Llama 3.1 (fastest, free tier available)"
    )

    # --- NVIDIA NIM ---
    NVIDIA_ENABLED: bool = True
    NVIDIA_API_KEY: Optional[str] = Field(
        default=None,
        description="NVIDIA API Key"
    )
    NVIDIA_MODEL: str = "nvidia/nemotron-3.5-lightning-30b-a3b"

    # --- Server ---
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    CORS_ORIGIN_REGEX: Optional[str] = Field(
        default=None,
        description="Optional regex for trusted preview origins, such as Vercel preview domains.",
    )

    # --- File Upload ---
    MAX_FILE_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: list[str] = [".png", ".jpg", ".jpeg", ".dcm"]

    model_config = {
        "env_file": ("../.env", ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


    @property
    def uploads_dir(self) -> str:
        return os.path.join(self.DATA_DIR, "uploads")

    @property
    def heatmaps_dir(self) -> str:
        return os.path.join(self.DATA_DIR, "heatmaps")

    @property
    def thumbnails_dir(self) -> str:
        return os.path.join(self.DATA_DIR, "thumbnails")

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    def has_llm_key(self) -> bool:
        """Check if any LLM API key is configured."""
        return bool(self.GEMINI_API_KEY or self.GROQ_API_KEY)

    def get_llm_provider_name(self) -> str:
        """Return the name of the first available LLM provider."""
        if self.GEMINI_API_KEY:
            return "gemini"
        elif self.GROQ_API_KEY:
            return "groq"
        return "template"

    def resolve_path(self, path: Optional[str]) -> Optional[str]:
        """Resolve relative repo paths regardless of the backend working directory."""
        if not path:
            return None
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(PROJECT_ROOT, path))



    @property
    def jeevansh_skin_cancer_path(self) -> Optional[str]:
        return self.resolve_path("models/jeevansh/skin_cancer.pth")

    @property
    def jeevansh_pneumonia_path(self) -> Optional[str]:
        return self.resolve_path("models/jeevansh/pneumonia.pth")

    @property
    def jeevansh_brain_tumor_path(self) -> Optional[str]:
        return self.resolve_path("models/jeevansh/brain_tumour.pt")

    @property
    def jeevansh_bone_fracture_path(self) -> Optional[str]:
        return self.resolve_path("models/jeevansh/fracture.pt")


# Singleton instance
settings = Settings()
