from src.schema import target_schema, TargetFormat
from src.config import Settings
from src.backends import InferenceBackendFactory


class NLPManager:
    def __init__(self, config: Settings):
        self.schema = target_schema
        self.config = config
        self.backend = InferenceBackendFactory().create(
            config.inference_backend, model_path=config.model_path
        )

    def generate_prompt(self, text: str) -> str:
        prompt = f"""
        <transcript>
            {text}
        </transcript>
        Based on the given voice transcript, convert it into a structured JSON, with the following JSON schema:
        <json_schema>
            {self.schema}
        </json_schema>
        You MUST only output the corresponding JSON based on the transcript provided.
        """
        return prompt


    def qa(self, texts: list[str], batch_size: int = 64) -> list[TargetFormat]:
        prompts = [self.generate_prompt(text) for text in texts]
        predictions = self.backend.infer(
            prompts, batch_size=batch_size
        )
        # Convert to JSON
        return [TargetFormat.model_validate_json(prediction) for prediction in predictions]
