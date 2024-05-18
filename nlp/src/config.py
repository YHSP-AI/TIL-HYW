from pathlib import Path
from typing import Union
from pydantic import Field
from pydantic_settings import BaseSettings

from src.schema import InferenceBackend

class Settings(BaseSettings):
    model_path: Union[Path, str] = "src/models/flan-t5-lora"
    inference_backend: InferenceBackend = InferenceBackend.HF
    batch_size: int = Field(64, ge=1)

def get_config():
    return Settings()
