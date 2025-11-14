#!/usr/bin/env python3
"""Basic example of benchmarking an entity extraction system."""

import argparse
import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from kebab import mskebab
from kebab.contracts.entity import Entity
from kebab.tasks.metrics.extraction.calculator import ExtractionOutput


class SimpleExtractor:
    """A simple extractor that uses regex patterns to extract entities from text."""

    def __init__(self):
        """Initialize the simple extractor."""
        uppercase_char = "[{}]".format("".join(chr(i) for i in range(sys.maxunicode) if chr(i).isupper()))
        name_pattern = rf"\"?{uppercase_char}[\w']+( {uppercase_char}[\w']+)*\"?"
        type_pattern = r"\w+( \w+){0,1}"
        self.pattern = (
            rf"(?P<name>{name_pattern})( \(.*\))* (is|are|was|were|refers to) (a|an|the) (?P<type>{type_pattern})"
        )
        self.name_exclusion_list = []  # List of names to exclude from extraction

    def _get_elbow_point(self, counter: Counter) -> int:
        """Identify an elbow point in the frequency distribution of items in the counter.

        Args:
            counter: A Counter object containing item frequencies.

        Returns:
            An integer representing the cliff point frequency.
        """
        previous_count = None
        previous_diff = None
        for idx, (_, count) in enumerate(counter.most_common()):
            if previous_count is not None:
                diff = previous_count - count
                if previous_diff is not None and diff < 0.01 * previous_diff:
                    return idx - 1
                previous_diff = diff
            previous_count = count
        return len(counter)

    def train(self, training_data: Iterable[ExtractionOutput]) -> None:
        """Train the extractor on the provided training data.

        Args:
            training_data: An iterable of ExtractionOutput objects representing the training data.
        """
        false_positive_names_counter = Counter()
        for item in training_data:
            text = item.document.data["text"]
            entity_names = set()
            for entity in item.entities:
                entity_names.update(entity.properties.get("name", []))
            matches = re.finditer(self.pattern, text)
            for match in matches:
                name = match.group("name")
                if name not in entity_names:
                    false_positive_names_counter[name] += 1

        for name, _ in false_positive_names_counter.most_common(self._get_elbow_point(false_positive_names_counter)):
            self.name_exclusion_list.append(name)

        print("Names to exclude:", self.name_exclusion_list)

    def generate_predictions(self, validation_data: Iterable[ExtractionOutput]) -> Iterable[list[Entity]]:
        """Generate example predictions for the given extraction task.

        Args:
            validation_data: An iterable of ExtractionOutput objects representing the validation data.

        Returns:
            An iterable of lists of Entity objects representing the predictions.
        """
        for item in validation_data:
            text = item.document.data["text"]
            matches = re.finditer(self.pattern, text)
            entities = []
            for id_, match in enumerate(matches):
                name = match.group("name")
                entity_type = match.group("type")
                if name not in self.name_exclusion_list:
                    properties = {
                        "name": [name],
                        "type": [entity_type],
                    }
                    entities.append(
                        Entity(
                            entity_id=str(id_),
                            properties=properties,
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
    parser.add_argument("--output_dir", type=Path, help="Optional output directory for results", default=Path("output"))
    args = parser.parse_args()
    repo_root = Path(__file__).parents[4]
    predictions_file = args.output_dir / "predictions.jsonl"
    # Initialize benchmark and task instance
    benchmark = mskebab.Benchmark(root_for_relative_paths=repo_root / "data")
    train_task_instance = benchmark.tasks_by_name["Extraction-ReDocRED-Train"]
    task_instance = benchmark.tasks_by_name["Extraction-ReDocRED-Test"]
    # Train the extractor predictions
    extractor = SimpleExtractor()
    extractor.train(train_task_instance.read_items())
    # Generate predictions and write to file
    predictions = extractor.generate_predictions(task_instance.read_items())
    task_instance.write_items(predictions_file, predictions)
    # Evaluate predictions
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_instance.evaluate(predictions_file, result_output_path=args.output_dir / "metrics.json")
    print(f"Benchmarking completed. Metrics saved to {args.output_dir}/metrics.json")
    print(f"You can view detailed extraction debug info Excel files in the {args.output_dir}/debug_output/ directory.")


if __name__ == "__main__":
    main()
