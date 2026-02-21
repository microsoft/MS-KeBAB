from itertools import islice
from pathlib import Path
from typing import cast

from kebab import mskebab
from kebab.contracts.entity import Entity
from kebab.tasks.linking import LinkingTask


def main() -> None:
    """Compute statistics of linking test pairs."""
    dataset_folder = Path(r"C:\Users\minka\Microsoft\Project Alexandria - Documents\Benchmark\Datasets")
    benchmark = mskebab.Benchmark(root_for_relative_paths=dataset_folder)
    task_instance = cast(LinkingTask, benchmark.tasks_by_name["Linking-REBEL-Test"])
    output_dir = Path(__file__).parents[0] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_list(e: Entity | list[Entity]) -> list[Entity]:
        """Ensure we have a list of entities."""
        return e if isinstance(e, list) else [e]

    def get_names(e: Entity | list[Entity]) -> set[str]:
        """Get all names from an entity or list of entities."""
        return {name for ent in make_list(e) for name in ent.properties["name"]}

    def get_values(e: Entity | list[Entity]) -> set[str]:
        """Get all values of all properties from an entity or list of entities."""
        return {value for ent in make_list(e) for values in ent.properties.values() for value in values}

    training_instance = benchmark.tasks_by_name["Linking-REBEL-Train"]
    print("Reading training items...(this takes about 1 minute)")
    training_values = {
        value
        for pair, label in islice(training_instance.read_items(), 10000000)
        for value in get_values(pair[0]).union(get_values(pair[1]))
    }
    print(f"Read {len(training_values)} training values.")

    # count the number of times each testing value appears in the training set
    testing_values = [get_values(pair[0]).union(get_values(pair[1])) for pair, label in task_instance.read_items()]
    overlap_percent = [
        sum(1 for value in values if value in training_values) / len(values) for values in testing_values
    ]

    # if False:
    #     shared_names = [
    #         get_names(pair[0]).intersection(get_names(pair[1])) for pair, label in task_instance.read_items()
    #     ]
    #     # count the number of times each shared name appears in the training set
    #     shared_counts = [sum(1 for s in training_names for name in names if name in s) for names in shared_names]
    #     shared_sizes = [len(names) for names in shared_names]
    #     print("Writing shared names statistics...")
    #     with open(output_dir / "shared_names.csv", "w", encoding="utf-8") as f:
    #         f.write("size,count,overlap\n")
    #         for size, count, overlap, names in zip(
    #             shared_sizes, shared_counts, overlap_counts, testing_names, strict=True
    #         ):
    #             line = ",".join(names)
    #             f.write(f"{size},{count},{overlap},{line}\n")

    not_entity_valued = {
        "name",
        "country",
        "point in time",
        "sport",
        "date of birth",
        "inception",
        "date of death",
        "publication date",
        "genre",
        "country of citizenship",
        "country of origin",
        "start time",
        "work period (start)",
        "end time",
        "dissolved, abolished or demolished date",
        "date of official opening",
        "time period",
        "continent",
    }

    def all_properties_not_entity_valued(entity: Entity | list[Entity]) -> bool:
        return all(all(prop in not_entity_valued for prop, values in e.properties.items()) for e in make_list(entity))

    pair_all_not_entity_valued = [
        all_properties_not_entity_valued(pair[0]) and all_properties_not_entity_valued(pair[1])
        for pair, label in task_instance.read_items()
    ]

    def any_nonname_property_contains(entity: Entity | list[Entity], names: set[str]) -> bool:
        return any(
            any(prop != "name" and any(name in values for name in names) for prop, values in e.properties.items())
            for e in make_list(entity)
        )

    pair_nonname_overlap = [
        any_nonname_property_contains(pair[0], get_names(pair[1]))
        or any_nonname_property_contains(pair[1], get_names(pair[0]))
        for pair, label in task_instance.read_items()
    ]

    predictions = [1.0 if s else 0.0 for s in pair_all_not_entity_valued]
    predictions_file = output_dir / "predictions.jsonl"
    task_instance.write_items(predictions_file, predictions)
    print(f"Wrote predictions to {predictions_file}")
    debugging_info_path = output_dir / "debugging_info.jsonl"
    with open(debugging_info_path, "w", encoding="utf-8") as f:
        for b in zip(pair_all_not_entity_valued, pair_nonname_overlap, overlap_percent, strict=True):
            s = f'{{"Not EVP": "{b[0]}", "Parent-child": "{b[1]}"'
            for i in range(10):
                fraction = i / 10
                s += f', "Overlap <= {fraction:.0%}": "{b[2] <= fraction}"'
            f.write(s + "}\n")

    # Compute metrics
    metrics = task_instance.evaluate(
        predictions_file,
        result_output_path=output_dir / "metrics.json",
        debugging_info_path=debugging_info_path,
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"You can view a detailed breakdown of results in the file {output_dir}/linking_predictions.tsv")


if __name__ == "__main__":
    main()
