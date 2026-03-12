from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NANORELAY_",
        extra="ignore"
    )

    port: int = 8080
    # backend_url: Optional[str] = None
    backend_config_path: str = "config.yaml"


settings = Settings()