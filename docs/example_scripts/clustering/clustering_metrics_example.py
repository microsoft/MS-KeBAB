from collections.abc import Iterable
from pathlib import Path
from typing import cast

from kebab import mskebab
from kebab.contracts.entity import Entity
from kebab.tasks.clustering import ClusteringTask
from kebab.utils.io_helpers import resolve_path
from scipy.cluster.hierarchy import DisjointSet


def main() -> None:
    """Run clustering metrics example."""
    dataset_folder = Path(__file__).parents[3] / "data"
    print(f"Using dataset folder: {dataset_folder}")
    if not resolve_path(dataset_folder / "REBEL" / "clustering").is_dir():
        print("You need to download the data first")
        return
    benchmark = mskebab.Benchmark(root_for_relative_paths=dataset_folder)
    task_instance = cast(ClusteringTask, benchmark.tasks_by_name["Clustering-REBEL-Test"])
    output_dir = Path(__file__).parents[0] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train a simple clusterer on the training data
    training_instance = cast(ClusteringTask, benchmark.tasks_by_name["Clustering-REBEL-Train"])
    print("Reading training items...")
    name_length_threshold = train(training_instance.read_items())
    print(f"Name length threshold: {name_length_threshold}")

    # Run the simple clusterer (cluster by name overlap) on the test data.
    # Note: the clusterer must not use any metadata, only properties.
    def filter_names(fragment: Entity, index: int) -> set[str]:
        """Get names longer than the threshold."""
        names = {name for name in fragment.properties["name"] if len(name) > name_length_threshold}
        if not names:
            names = {f"__no_valid_names_{index}__"}
        return names

    names_of_fragment = [filter_names(fragment, i) for i, (fragment, label) in enumerate(task_instance.read_items())]
    # Merge sets by name overlap
    ds = DisjointSet({name for names in names_of_fragment for name in names})
    first_names = [next(iter(names)) for names in names_of_fragment]
    for first_name, names in zip(first_names, names_of_fragment, strict=True):
        for name in names:
            ds.merge(first_name, name)
    predictions = [ds[first_name] for first_name in first_names]
    predictions_file = output_dir / "predictions.txt"
    task_instance.write_items(predictions_file, predictions)
    print(f"Wrote predictions to {predictions_file}")

    # Compute metrics
    metrics = task_instance.evaluate(
        predictions_file,
        result_output_path=output_dir / "metrics.json",
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


def train(data: Iterable[tuple[Entity, str | None]]) -> int:
    """Simple training for clustering by name overlap."""
    labels_of_name = {}
    for entity, label in data:
        for name in entity.properties["name"]:
            if name not in labels_of_name:
                labels_of_name[name] = set()
            if label is not None:
                labels_of_name[name].add(label)
    average_label_count_by_name_length = {}
    for name, labels in labels_of_name.items():
        length = len(name)
        total_count, number_of_names = average_label_count_by_name_length.get(length, (0, 0))
        label_count = len(labels)
        average_label_count_by_name_length[length] = (total_count + label_count, number_of_names + 1)

    # Find the name length threshold that excludes most ambiguous names
    name_length_threshold = 0
    average_label_threshold = 1.5
    for length, (total_count, number_of_names) in average_label_count_by_name_length.items():
        if number_of_names > 0:
            average_label_count = total_count / number_of_names
            if average_label_count > average_label_threshold and length > name_length_threshold:
                name_length_threshold = length

    return name_length_threshold


if __name__ == "__main__":
    main()
