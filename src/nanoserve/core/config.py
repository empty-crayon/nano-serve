from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NANOSERVE_",
        extra="ignore"
    )

    host: str = "127.0.0.1"   # bind to localhost; nginx (port 8780) fronts the public interface
    port: int = 8765           # must match upstream port in monitoring/nginx-gateway-lb.conf
    backend_config_path: str = "config.yaml"


settings = Settings()