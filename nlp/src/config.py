from pathlib import Path
from typing import Union
from pydantic import Field
from pydantic_settings import BaseSettings

from src.schema import InferenceBackend

class Settings(BaseSettings):
    model_path: Union[Path, str] = "src/models/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf"
    inference_backend: InferenceBackend = InferenceBackend.LCPP
    batch_size: int = Field(64, ge=1)

def get_config():
    return Settings()
