#!/usr/bin/env python3
"""Basic example of benchmarking an entity extraction system."""

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

from kebab import mskebab
from kebab.contracts.entity import Entity

import nltk


def generate_predictions(task_instance: mskebab.Task) -> Iterable[list[Entity]]:
    """Generate example predictions for the given extraction task.

    Args:
        task_instance: The extraction task instance for which to generate predictions.
    Returns:
        Path to the generated predictions JSONL file.
    """
    uppercase_char = "[{}]".format("".join(chr(i) for i in range(sys.maxunicode) if chr(i).isupper()))
    name_pattern = r"\"?{upper}[\w']+( {upper}[\w']+)*\"?".format(upper=uppercase_char)
    type_pattern = r"\w+( \w+){0,1}"
    pattern = r"(?P<name>{name_pattern})( \(.*\))* (is|are|was|were|refers to) (a|an|the) (?P<type>{type_pattern})".format(
        name_pattern=name_pattern, type_pattern=type_pattern
    )
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    for item in task_instance.read_items():
        text = item.document.data["text"]
        matches = re.finditer(pattern, text)
        entities = []
        for id, match in enumerate(matches):
            name = match.group("name")
            entity_type = match.group("type")
            if name not in ["It", "They", "He", "She", "This", "That", "There", "Those"]:
                entities.append(
                    Entity(
                        entity_id=str(id),
                        properties={
                            "name": [name],
                            "type": [entity_type],
                        },
                    )
                )

        yield entities

def main():
    """Entrypoint function to run the extraction benchmark."""
    # This assumes that the evaluation dataset is located in the `data/` directory at the repo root
    # To download the dataset into the data/ directory, run from repo root:
    # ./scripts/dataset/download_and_preprocess_redocred.sh
    parser = argparse.ArgumentParser(
        description="Example extraction benchmark runner using an existing predictions file."
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Optional output directory for results", default=Path("output_dir")
    )
    args = parser.parse_args()
    repo_root = Path(__file__).parents[4]
    predictions_file = Path("predictions.jsonl")
    # Initialize benchmark and task instance
    benchmark = mskebab.Benchmark(config_path=repo_root / "kebab" / "configs" / "tasks.json", root_for_relative_paths=repo_root / "data")
    task_instance = benchmark.tasks_by_name["Extraction-ReDocRED-Test"]
    # Generate predictions and write to file
    task_instance.write_items(predictions_file, generate_predictions(task_instance))
    # Evaluate predictions
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_instance.evaluate(predictions_file, result_output_path=args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()