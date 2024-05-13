from enum import Enum

from pydantic import BaseModel, Field

class InferenceBackend(str, Enum):
    HF = "transformers"
    LCPP = "llamacpp"
    EXLV2 = "exllamav2"

class InferenceInput(BaseModel):
    key: int
    transcript: str

class InferenceRequest(BaseModel):
    instances: list[InferenceInput]

class TargetFormat(BaseModel):
    target: str = Field(description="The enemy to target, including its description from the transcript (e.g color). Do not paraphrase from the original transcript.")
    tool: str
    heading: str = Field(pattern="[0-9][0-9][0-9]")

target_schema = TargetFormat.model_json_schema()

print(target_schema)