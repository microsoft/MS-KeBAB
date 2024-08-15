# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pyright: reportOptionalMemberAccess=false
# ruff: noqa: PYI024, ANN201, ANN001, FA102, A001, A002, D417, D103, TRY401
import json
import re
from collections import Counter, defaultdict, namedtuple

from kebab.extraction.metrics.metric_helpers import Entity


Property = namedtuple("Property", ["name", "is_set"])
Metrics = namedtuple("Metrics", ["counts", "hallucinations", "generated_property_counts", "target_property_counts"])


def normalize_value(value: str) -> str:
    """Replaces all space like symbols to spaces, removes leading and trailing spaces and double spaces."""
    return re.sub(r"\s+", " ", value.strip()).lower()


def get_counts(counts_list: list[dict[str, float]]) -> Counter:
    """Creates a dictionary of counts of each given property or type."""
    counters = [Counter(d) for d in counts_list]
    total_property_counts = sum(counters, Counter())
    return total_property_counts


def get_hallucination_fraction(hallucinations_list: list[dict[str, float]]) -> dict[str, float]:
    """Computes the hallucination fraction for each property."""
    counts = get_counts(hallucinations_list)
    hallucination_fraction = {}
    for key in counts:
        if not key.endswith("_count"):
            hallucination_fraction[key] = counts[key] / counts[key + "_count"]

    return hallucination_fraction


def get_property_values(entity: Entity, properties: list[Property]) -> set[str]:
    """Returns the values of the given properties for the given entity as a unified set of values."""
    values = set()
    for property, is_set in properties:
        if property not in entity:
            continue

        if is_set:
            for value in entity[property]:
                values.add(normalize_value(str(value)))
        else:
            values.add(normalize_value(str(entity[property])))

    return values


def compute_hallucination(generated_output: list[dict], target_entities: list[dict], context: str) -> dict[str, int]:
    """Compute hallucination fraction for properties that should not be inferred."""
    context_lower = normalize_value(context)
    properties_for_exact_match = {
        "name",
        "alternative names",
        "acronym",
        "located in",
        "location",
        "compatible with",
        "provided by",
        "part of",
        "url",
        "related to",
        "related_entity",
        "owner",
        "depends on",
        "email",
    }

    generated_properties_for_exact_match = defaultdict(list)

    def calculate_hallucination(dataset: list[dict], suffix: str):
        for entity in dataset:
            for property in properties_for_exact_match:
                if isinstance(entity, dict) and property in entity and isinstance(entity[property], str):
                    generated_properties_for_exact_match[property + suffix].append(normalize_value(entity[property]))

    calculate_hallucination(generated_output, "_hallucination")
    calculate_hallucination(target_entities, "_hallucination_target")

    property_hallucination = {}
    for prop, prop_value_list in generated_properties_for_exact_match.items():
        hallucinated_count = 0
        for prop_value in prop_value_list:
            if prop_value not in context_lower:
                hallucinated_count += 1
        property_hallucination[prop] = hallucinated_count
        property_hallucination[prop + "_count"] = len(prop_value_list)

    return property_hallucination


def evaluate_metrics(data: dict, output_json_already_processed=False, tokenizer=None, logger=None) -> Metrics:
    """
    Evaluate the generated text based on various metrics.

    Args:
        data (dict): A dictionary containing the context, target, and generated output.

    Returns:
        counts: similar as above, but now the absolute counts instead of
            fractions, such that we can compute total statistics for the whole
            dataset afterwards.
        hallucinations: A dictionary containing the hallucination number for each property. The hallucination
            fraction is calculated later based on all data.
            - <property>_hallucination: The number of generated property values
              that are not present in the input context.
            - <property>_hallucination_target: The number of target
              property values that are not present in the input context (ground truth
              hallucination fraction).
    """
    context = normalize_value(data["context"]) if "context" in data else normalize_value(data["raw_text"])
    target_entities = data["target"]
    if output_json_already_processed:
        generated_output = data["generated_output"]
    else:
        try:
            input_str = data["generated_output"]
            if tokenizer:
                if tokenizer.bos_token is not None and input_str.startswith(tokenizer.bos_token):
                    input_str = input_str[len(tokenizer.bos_token) :]
                if tokenizer.pad_token is not None and input_str.startswith(tokenizer.pad_token):
                    input_str = input_str[len(tokenizer.pad_token) :]
                if tokenizer.eos_token is not None and input_str.endswith(tokenizer.eos_token):
                    input_str = input_str[: -len(tokenizer.eos_token)]
            generated_output = json.loads(input_str)
        except Exception as e:
            logger.exception(e)
            raise

    for entity in generated_output:
        if not isinstance(entity, dict):
            generated_output.remove(entity)

    generated_names = {
        normalize_value(str(entity["name"]))
        for entity in generated_output
        if (isinstance(entity, dict) and "name" in entity)
    }

    target_names = {
        normalize_value(str(entity["name"]))
        for entity in target_entities
        if (isinstance(entity, dict) and "name" in entity)
    }

    counts = {
        "generated_entities": len(generated_names),
        "target_entities": len(target_names),
        "document_count": 1,
    }

    property_hallucination = compute_hallucination(generated_output, target_entities, context)

    # count the number of property values for each property are extracted
    generated_property_counts = count_property_keys(generated_output)
    target_property_counts = count_property_keys(target_entities)

    return Metrics(counts, generated_property_counts, target_property_counts, property_hallucination)


def count_property_keys(entities: list[Entity]) -> dict[str, int]:
    property_counts = defaultdict(int)
    for entity in entities:
        for property in entity:
            if isinstance(entity[property], str):
                property_counts[property] += 1
            elif isinstance(entity[property], list):
                for value in entity[property]:
                    if isinstance(value, str):
                        property_counts[property] += 1

    return property_counts
