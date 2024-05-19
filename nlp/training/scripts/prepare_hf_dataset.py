# Script to convert jsonl data to a HF dataset
import typer
import json

from pathlib import Path
from typing_extensions import Annotated

import nltk

from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")

label_to_class = {
    "O": 0,
    "B-TAR": 1,
    "I-TAR": 2,
    "B-TOOL": 3,
    "I-TOOL": 4,
    "B-DIR": 5,
    "I-DIR": 6,
}

number_to_phonetic = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "niner",
}

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


# https://huggingface.co/learn/nlp-course/chapter7/2?fw=pt#processing-the-data
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def process_example(example):
    # words still need to pass through subword tokenization
    transcript: str = example["transcript"]
    transcript = transcript.strip().lower()
    tokens = word_tokenize(transcript)
    # Initalize all none classes
    label_names = ["O"] * len(tokens)

    tool: str = example["tool"]
    heading: str = example["heading"]
    target: str = example["target"]

    # Function to find the next start index of a tokenized entity in tokenized text without overlap
    def find_start_index(entity_tokens, tokens, label_names):
        entity_length = len(entity_tokens)
        for i in range(len(tokens) - entity_length + 1):
            if tokens[i : i + entity_length] == entity_tokens:
                # Check for overlap
                overlap = False
                for j in range(i, i + entity_length):
                    if label_names[j] != "O":
                        overlap = True
                        break
                if not overlap:
                    return i
        return -1

    # Function to label tokens, ensuring no overlap
    def label_tokens(entity_tokens, label_prefix):
        start_idx = find_start_index(entity_tokens, tokens, label_names)
        if start_idx != -1:
            # Label tokens
            label_names[start_idx] = f"B-{label_prefix}"
            for i in range(1, len(entity_tokens)):
                label_names[start_idx + i] = f"I-{label_prefix}"

    tool_tokens = word_tokenize(tool)
    target_tokens = word_tokenize(target)
    heading_tokens = [number_to_phonetic[x] for x in heading]

    label_tokens(tool_tokens, "TOOL")
    label_tokens(target_tokens, "TAR")
    label_tokens(heading_tokens, "DIR")

    # Get raw labels
    inputs = tokenizer(tokens, is_split_into_words=True, truncation=True)

    word_ids = inputs.word_ids()
    new_labels = align_labels_with_tokens(
        [label_to_class[x] for x in label_names], word_ids
    )

    inputs["labels"] = new_labels


    return inputs


def main(
    raw_data_path: Annotated[
        Path, typer.Argument(dir_okay=False, exists=True)
    ] = "./data/train.jsonl",
    output_dir: Annotated[Path, typer.Argument()] = "./data/processed/train",
):
    # Create HF dataset (with optional train val split (?))
    dataset = load_dataset(
        "json", data_files=str(raw_data_path), split="train"
    )

    # Process each example
    # Using the labels as GT, find the location of the target, tool and heading
    # Generate respective NER labels B-TAR, I-TAR, B-TOOL, I-TOOL, B-DIR, I-DIR
    # Generate the following
    dataset = dataset.map(process_example)

    typer.echo("Dataset Created, Preview:")
    for i in range(min(len(dataset), 5)):
        typer.echo(dataset[i])

    # Save to output dir
    typer.echo(f"Saving to directory: {output_dir}")
    dataset.save_to_disk(str(output_dir))


if __name__ == "__main__":
    typer.run(main)
