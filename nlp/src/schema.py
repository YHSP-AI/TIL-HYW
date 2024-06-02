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
    target: str = Field(description="The verbatim description of the enemy to be targeted, as stated in the operator's transcript.")
    tool: str = Field(description="The verbatim name of the weapon or tool to be used against the target, as stated in the operator's transcript.")
    heading: str = Field(pattern="[0-9][0-9][0-9]", description="The 3-digit compass bearing (between 000 and 360) indicating the direction of the target, specified as a number.")


target_schema = TargetFormat.model_json_schema()
