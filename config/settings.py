from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openrouter_api_key: str | None = None
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
    api_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    embed_url: str = "https://openrouter.ai/api/v1/embeddings"
    frontend_url: str = "http://localhost:3000"
    ws_base_url: str = "ws://localhost:8000"

    class Config:
        env_file = ".env"
        case_sensitive = False


