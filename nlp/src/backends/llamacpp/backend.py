import json
from pathlib import Path
from typing import List

from llama_cpp import Llama, LogitsProcessorList
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from huggingface_hub import hf_hub_download
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.llamacpp import (
    build_llamacpp_logits_processor,
    build_token_enforcer_tokenizer_data,
)

from src.backends.base import NLPInferenceBackend
from src.schema import target_schema, TargetFormat


class LCPPInferenceBackend(NLPInferenceBackend):
    def __init__(self, model_path: str):
        # Check if actually a path
        if Path(model_path).exists():
            self.path = model_path
        else:
            # See if we can download from HF
            repo_id, filename = model_path.rsplit("/", maxsplit=1)
            downloaded_path = hf_hub_download(
                repo_id=repo_id, filename=filename
            )
            self.path = downloaded_path

        # Use LM Format Enforcer to force output
        # to conform to JSON schema
        self.llm = Llama(
            model_path=self.path,
            n_gpu_layers=-1,
            draft_model=LlamaPromptLookupDecoding(),
            n_ctx=2048,
        )
        self.tokenizer_data = build_token_enforcer_tokenizer_data(self.llm)
        self.schema = JsonSchemaParser(
            target_schema, num_consecutive_whitespaces=6
        )
        self.logits_processors = LogitsProcessorList(
            [build_llamacpp_logits_processor(self.tokenizer_data, self.schema)]
        )

    def infer(self, inputs: List[List[dict]], **kwargs) -> List[str]:
        # NOTE: LCPP Python bindings don't support batched inference yet
        predictions = []
        failed = 0
        for prompt in inputs:
            # https://github.com/NousResearch/Hermes-Function-Calling/blob/main/jsonmode.py
            # TODO: recursively attempt to resolve JSON validation errors on model end
            attempt = 1
            while True:
                try:
                    output = self.llm.create_chat_completion(
                        messages=prompt,
                        logits_processor=self.logits_processors,
                        temperature=0
                    )
                    text: str = output["choices"][0]["message"]["content"]
                    TargetFormat.model_validate_json(text)
                    break
                except Exception as e:
                    print(e)
                    attempt -= 1
                    if attempt == 0:
                        print(text)
                        text = json.dumps(
                            {"tool": "", "heading": "000", "target": ""}
                        )
                        failed += 1
                        print("Failed to generate")
                        break
            predictions.append(text)
        print("Failed Preds: ", failed)
        return predictions
