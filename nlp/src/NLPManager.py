from textwrap import dedent
from typing import List
from json import dumps

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
        self.system_prompt = f"""
You are an AI assistant integrated into a critical military defense system for engaging enemy targets. Your task is to convert voice transcripts from turret operators into precisely formatted JSON commands to be sent to the turrets. Incorrect outputs could result in unintended casualties, so it is essential that you strictly follow the specified JSON schema and formatting rules.

When processing a transcript, use the verbatim wording from the operator whenever possible. The only exception is the 'heading' field, which must always be a 3-digit number representing the compass bearing of the target (e.g., 051).

# Examples
Here are some examples of valid operator transcripts and their expected JSON outputs:

## Example 1
### Input
"Turret Alpha, deploy anti-air artillery to intercept the grey, blue, and yellow fighter plane at heading zero eight five."
### Output
{dumps({"tool": "anti-air artillery", "heading": "085", "target": "grey, blue, and yellow fighter plane"}, indent=2)}

## Example 2
### Input
"Turret Bravo, fire TOW missiles at enemy T-90SM tank heading two seven niner."
### Output
{dumps({"tool": "TOW missiles", "heading": "279", "target": "T-90SM tank"}, indent=2)}

# Output Instructions
Answer in valid JSON. Here is the relevant JSON schema you must adhere to:

<schema>
{dumps(target_schema, indent=2)}
</schema>

Your outputs must strictly adhere to the provided JSON schema, ensuring all fields exist, and that the JSON is valid and complete (do not output a partially complete/unclosed JSON string). This is a matter of life and death, so precision is paramount.
                            """

    def generate_prompt(self, text: str) -> List[dict]:
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": dedent(
                    f"""
                # Input
                {text}
                # Output
                """
                ),
            },
        ]

    def qa(self, texts: List[str], batch_size: int = 64) -> List[TargetFormat]:
        prompts = [self.generate_prompt(text) for text in texts]
        predictions = self.backend.infer(prompts, batch_size=batch_size)
        # Convert to JSON
        return [
            TargetFormat.model_validate_json(prediction)
            for prediction in predictions
        ]
