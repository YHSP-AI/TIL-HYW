import json
from pathlib import Path
from typing import List
from string import punctuation

from torch.cuda import is_available
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    TokenClassificationPipeline,
)
from peft import PeftModel, PeftConfig
from huggingface_hub import hf_hub_download

from src.backends.base import NLPInferenceBackend


class HFInferenceBackend(NLPInferenceBackend):
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
        self.phonetics_to_digits = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "niner": "9",
            "nine": "9",  # just in case
        }
        self.label2id = {
            "O": 0,
            "B-TAR": 1,
            "I-TAR": 2,
            "B-TOOL": 3,
            "I-TOOL": 4,
            "B-DIR": 5,
            "I-DIR": 6,
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        config = PeftConfig.from_pretrained(self.path)
        self.device = "cuda" if is_available() else "cpu"
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            label2id=self.label2id,
            id2label=self.id2label,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path
        )
        self.model = PeftModel.from_pretrained(self.model, self.path)
        self.model.eval()
        self.ner: TokenClassificationPipeline = pipeline(
            "token-classification",
            self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="first",
        )

    def infer(self, inputs: List[str], **kwargs) -> List[str]:
        processed_inputs = [text.lower() for text in inputs]
        outputs = self.ner(processed_inputs)
        predictions = []
        for output, text in zip(outputs, inputs):
            pred = {"tool": "", "heading": "000", "target": ""}
            print(output)
            for entity_grp in ("TOOL", "TAR", "DIR"):
                min_idx = len(text)
                max_idx = -1
                for result in output:
                    if result["entity_group"] != entity_grp:
                        continue
                    min_idx = result["start"]
                    max_idx = result["end"]
                    break
                span = text[min_idx:max_idx].strip().strip(punctuation)
                print(f"{entity_grp}: {span}")

                if entity_grp == "TOOL":
                    pred["tool"] = span
                elif entity_grp == "DIR":
                    # Convert to 3 digit
                    heading = "".join(
                        [self.phonetics_to_digits[x.lower()] for x in span.split()]
                    )
                    if len(heading) != 3:
                        heading = "000"
                    pred["heading"] = heading
                else:
                    pred["target"] = span
            predictions.append(json.dumps(pred))

        return predictions
