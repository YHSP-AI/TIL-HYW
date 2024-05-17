from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class InferenceBackend(str, Enum):
    HF = "transformers"
    LCPP = "llamacpp"
    EXLV2 = "exllamav2"
    VLLM = "vllm"


class InferenceInput(BaseModel):
    key: int
    transcript: str


class InferenceRequest(BaseModel):
    instances: List[InferenceInput]


class TargetFormat(BaseModel):
    target: str
    tool: str
    heading: str = Field(pattern="[0-9][0-9][0-9]")


target_schema = TargetFormat.model_json_schema()
