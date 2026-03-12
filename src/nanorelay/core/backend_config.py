import yaml
from pydantic import BaseModel

class BackendEntry(BaseModel):
    type: str
    url: str
    timeout: float 

class BackendConfig(BaseModel):
    default: str
    backends: list[BackendEntry]

def load_backend_config(path: str) -> BackendConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    config = BackendConfig(**data)
    _validate_backend_config(config)

    return config

def _validate_backend_config(config: BackendConfig):
    if config.default not in [backend.type for backend in config.backends]:
        raise ValueError(f"Default backend '{config.default}' is not defined in backends")
    
    # check for duplicate type values across backends
    types = set()
    for entry in config.backends:
        if entry.type in types:
            raise ValueError(f"Duplicate backend type '{entry.type}' found in backends")
        types.add(entry.type)
    