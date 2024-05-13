import typer

from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

def main(
    dataset: Annotated[Path, typer.Option()],
    train_split: Annotated[float, typer.Option(min=0, max=1)] = 0.8,
    output_dir: Annotated[Optional[Path], typer.Option()] = None
):
    if not dataset.is_file():
        print("Dataset not supplied")
        raise typer.Abort()
    if output_dir is None:
        output_dir = Path('.')
    if not output_dir.is_dir():
        print("Output dir needs to be a file directory")
        raise typer.Abort()


    # Load in JSONL file
    with open(dataset, "r") as file:
        lines = file.readlines()

    # Determine split
    pivot = int(len(lines) * train_split)

    train = lines[:pivot]
    validation = lines[pivot:]

    print(f"Train Size: {len(train)}")
    print(f"Validation Size: {len(validation)}")

    print("First five examples")
    for i in range(min(len(train), 5)):
        print(train[i])

    # Output
    for filename, split in (('train.jsonl', train), ('val.jsonl', validation)):
        with open(output_dir / filename, 'w') as file:
            file.writelines(split)



if __name__ == "__main__":
    typer.run(main)