from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from peft import PeftModel, PeftConfig
from torch.cuda import is_available

def cache_model(model_path):
    phonetics_to_digits = {
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
    label2id = {
            "O": 0,
            "B-TAR": 1,
            "I-TAR": 2,
            "B-TOOL": 3,
            "I-TOOL": 4,
            "B-DIR": 5,
            "I-DIR": 6,
        }
    id2label = {v: k for k, v in label2id.items()}

    config = PeftConfig.from_pretrained(model_path)
    device = "cuda" if is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            label2id=label2id,
            id2label=id2label,
    )
    tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path
        )
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    ner = pipeline(
            "token-classification",
            model,
            tokenizer=tokenizer,
            aggregation_strategy="first",
    )


if __name__ == "__main__":
    model_path = "src/models/flan-t5-base-lora-rslora-v1.1"
    cache_model(model_path)