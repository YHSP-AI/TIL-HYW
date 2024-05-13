from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

from src.schema import InferenceBackend

class Settings(BaseSettings):
    model_path: Path | str = "src/models/Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf"
    inference_backend: InferenceBackend = InferenceBackend.LCPP
    batch_size: int = Field(64, ge=1)

def get_config():
    return Settings()
