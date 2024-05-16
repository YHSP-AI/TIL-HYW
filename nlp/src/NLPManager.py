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

    def generate_prompt(self, text: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": f"""
                            You are a helpful assistant that outputs in JSON. 
                            Your goal is to process incoming instructions transcribed 
                            from a defence turret operator, and convert it to a JSON which 
                            represents a command to be sent to the turret. Try to avoid paraphrasing
                            and instead use exact words from the transcript when generating
                            the JSON, with the exception of the heading which should always
                            be three digits (e.g 051).

                            The expected JSON schema is as follows:
                            <json_schema>
                                {self.schema}
                            </json_schema>

                            <example>
                                <transcript>
                                    Turret Alpha, deploy anti-air artillery to intercept the grey, blue, and yellow fighter plane at heading one eight five.
                                </transcript>
                                <expected_output>
                                    {{"tool": "anti-air artillery", "heading": "185", "target": "grey, blue, and yellow fighter plane"}}
                                </expected_output>
                            </example>
                            """,
            },
            {
                "role": "user",
                "content": f"""
                <transcript>
                    {text}
                </transcript>
                Based on the given voice transcript, convert it into a structured JSON
                """,
            },
        ]

    def generate_prompt_old(self, text: str) -> str:
        prompt = f"""
        <|im_start|>system

        <json_schema>
            {self.schema}
        </json_schema>

        You MUST only output the corresponding JSON based on the transcript provided.

        <|im_end|>
        <|im_start|>user
        Based on the given voice transcript, convert it into a structured JSON, with the following JSON schema:
        <transcript>
            {text}
        </transcript>
        <|im_end|>
        """
        return prompt

    def qa(self, texts: list[str], batch_size: int = 64) -> list[TargetFormat]:
        prompts = [self.generate_prompt(text) for text in texts]
        predictions = self.backend.infer(prompts, batch_size=batch_size)
        # Convert to JSON
        return [
            TargetFormat.model_validate_json(prediction)
            for prediction in predictions
        ]
