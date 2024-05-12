from src.schema import target_schema


class NLPManager:
    def __init__(self):
        self.schema = target_schema

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
        return []
