from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openrouter_api_key: str | None = None
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
