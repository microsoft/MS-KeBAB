import math
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import cast

from kebab import mskebab
from kebab.contracts.entity import Entity
from kebab.tasks.linking import LinkingTask


def train(data: Iterable[tuple[tuple[Entity, Entity], bool]]) -> tuple[float, float]:
    """Simple training for linking by name overlap."""
    positive_with_overlap = 1
    positive_without_overlap = 1
    negative_with_overlap = 1
    negative_without_overlap = 1
    for pair, label in data:
        shared_name_count = len(set(pair[0].properties["name"]).intersection(pair[1].properties["name"]))
        if label:
            if shared_name_count > 0:
                positive_with_overlap += 1
            else:
                positive_without_overlap += 1
        else:
            if shared_name_count > 0:
                negative_with_overlap += 1
            else:
                negative_without_overlap += 1
    print(f"Positive with overlap: {positive_with_overlap}")
    print(f"Positive without overlap: {positive_without_overlap}")
    print(f"Negative with overlap: {negative_with_overlap}")
    print(f"Negative without overlap: {negative_without_overlap}")
    return (
        math.log(positive_without_overlap / negative_without_overlap),
        math.log(positive_with_overlap / negative_with_overlap),
    )


def main() -> None:
    """Run linking metrics example."""
    dataset_folder = Path(__file__).parents[3] / "data"
    print(f"Using dataset folder: {dataset_folder}")
    if not (dataset_folder / "REBEL").is_dir():
        print("You need to download the data first")
        return
    benchmark = mskebab.Benchmark(root_for_relative_paths=dataset_folder)
    task_instance = cast(LinkingTask, benchmark.tasks_by_name["Linking-REBEL-Test"])
    output_dir = Path(__file__).parents[0] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train a simple linker on the training data
    training_instance = benchmark.tasks_by_name["Linking-REBEL-Train"]
    print("Reading training items...")
    (log_odds_without_overlap, log_odds_with_overlap) = train(islice(training_instance.read_items(), 10000))
    print("Done.")

    # Run the simple linker (link by name overlap) on the test data
    shared_names_count = [
        len(set(pair[0].properties["name"]).intersection(pair[1].properties["name"]))
        for pair, label in task_instance.read_items()
    ]
    predictions = [log_odds_with_overlap if count > 0 else log_odds_without_overlap for count in shared_names_count]
    predictions_file = output_dir / "predictions.jsonl"
    task_instance.write_items(predictions_file, predictions)
    print(f"Wrote predictions to {predictions_file}")
    debugging_info_path = output_dir / "debugging_info.jsonl"
    with open(debugging_info_path, "w", encoding="utf-8") as f:
        for count in shared_names_count:
            f.write(f'{{"shared name count": {count}}}\n')

    # Compute metrics
    metrics = task_instance.evaluate(
        predictions_file,
        output_dir=output_dir,
        result_output_path=output_dir / "metrics.json",
        debugging_info_path=debugging_info_path,
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
