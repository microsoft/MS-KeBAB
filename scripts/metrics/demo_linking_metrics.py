import json
import math
import shutil
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from kebab.contracts.entity import Entity
from kebab.tasks.linking import LinkingTask

def main() -> None:
    dataset_folder = Path(r"C:\Users\minka\Microsoft\Project Alexandria - Documents\Benchmark\Datasets\REBEL\Linking\Test")
    entity_pairs_path = dataset_folder / "rebel_linking_dataset.jsonl"
    labels_path = dataset_folder / "rebel_linking_ground_truth.jsonl"
    schema_file_path = dataset_folder / "property_schema.json"
    task_instance = LinkingTask(
        "Linking-Metrics-Test", str(entity_pairs_path), str(schema_file_path), str(labels_path)
    )

    items = list(task_instance.read_items())
    boolean_labels = [True for _, predicted_boolean in items if predicted_boolean is not None]
    output_dir = Path(__file__).parents[1] / "output" / "linking"
    output_dir.mkdir(parents=True, exist_ok=True)
    boolean_labels_output_file_path = output_dir / "boolean_labels.jsonl"
    task_instance.write_items(boolean_labels_output_file_path, boolean_labels)
    metrics = task_instance.evaluate(boolean_labels_output_file_path) #, eval_result_path = output_dir / "metrics.json") #, output_dir = output_dir)
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()