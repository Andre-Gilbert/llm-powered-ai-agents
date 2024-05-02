"""Global settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Class that implements the global settings."""

    CLIENT_ID: str
    CLIENT_SECRET: str
    AUTH_URL: str
    API_BASE: str

    API_MAX_RETRIES: int = 15
    API_MIN_RETRY_TIMEOUT_SECONDS: int = 3
    API_MAX_RETRY_TIMEOUT_SECONDS: int = 60
    API_REQUEST_TIMEOUT_SECONDS: int = 300
    API_ACCESS_TOKEN_EXPIRY_MINUTES: int = 60

    GPT_4_REQUEST_LIMIT_PER_MINUTE: int
    GPT_35_REQUEST_LIMIT_PER_MINUTE: int
    TEXT_ADA_002_REQUEST_LIMIT_PER_MINUTE: int

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
