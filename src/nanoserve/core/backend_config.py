import yaml
from pydantic import BaseModel


class BackendConfig(BaseModel):
    backend_url: str
    timeout: float = 120
    echo_timeout: float = 10


def load_backend_config(path: str) -> BackendConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return BackendConfig(**data)