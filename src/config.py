"""Configuration management for the legal contract chatbot."""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Claude API Configuration
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    claude_model: str = Field(default="claude-3-5-sonnet-20241022", env="CLAUDE_MODEL")
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    
    # Vector Database Configuration
    vector_db_type: str = Field(default="chromadb", env="VECTOR_DB_TYPE")
    chroma_persist_directory: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_enabled: bool = Field(default=False, env="REDIS_ENABLED")
    
    # Performance Configuration
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    batch_size: int = Field(default=10, env="BATCH_SIZE")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Monitoring Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    alert_threshold_latency_ms: int = Field(default=200, env="ALERT_THRESHOLD_LATENCY_MS")
    
    # Streamlit Configuration
    streamlit_server_port: int = Field(default=8501, env="STREAMLIT_SERVER_PORT")
    streamlit_server_address: str = Field(default="0.0.0.0", env="STREAMLIT_SERVER_ADDRESS")
    
    # Contract Data
    contract_json_path: str = Field(default="./RFP_parsed.json", env="CONTRACT_JSON_PATH")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
