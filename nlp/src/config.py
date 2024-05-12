from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

from src.schema import InferenceBackend

class Settings(BaseSettings):
    model_path: Path | str = ""
    inference_backend: InferenceBackend = InferenceBackend.LCPP
    batch_size: int = Field(64, ge=1)

def get_config():
    return Settings()
