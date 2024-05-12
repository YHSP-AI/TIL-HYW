from pathlib import Path

from llama_cpp import Llama, LogitsProcessorList
from huggingface_hub import hf_hub_download
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor, build_token_enforcer_tokenizer_data

from src.backends.base import NLPInferenceBackend
from src.schema import target_schema

class LCPPInferenceBackend(NLPInferenceBackend):
    def __init__(self, model_path: str):
        # Check if actually a path
        if Path(model_path).exists():
            self.path = model_path
        else:
            # See if we can download from HF
            repo_id, filename = model_path.rsplit("/", maxsplit=1)
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
            self.path = downloaded_path

        # Use LM Format Enforcer to force output
        # to conform to JSON schema
        self.llm = Llama(model_path=self.path)
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.llm)
        self.schema = JsonSchemaParser(target_schema)
        self.logits_processors = LogitsProcessorList([
            build_llamacpp_logits_processor(self.tokenizer_data, self.schema)
        ])

    def infer(self, inputs: list[str], **kwargs) -> list[str]:
        # NOTE: LCPP Python bindings don't support batched inference yet
        predictions = []
        for prompt in inputs:
            output = self.llm(prompt, logits_processor=self.logits_processors, max_tokens=256)
            text: str = output["choices"][0]["text"]
            predictions.append(text)
        return predictions
