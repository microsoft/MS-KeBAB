# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pyright: reportPossiblyUnboundVariable=false
# ruff: noqa: D103, ANN001, UP015, C401, SIM118
import json

from sentence_transformers import SentenceTransformer

from kebab.extraction.metrics.bipartite_matching import (
    compute_bipartite_metrics,
    compute_scores_across_documents,
    compute_umatched_statistics,
)
from kebab.extraction.metrics.metric_helpers import MetricsAccumulator
from kebab.extraction.metrics.property_statistics import evaluate_metrics, get_counts, get_hallucination_fraction
from kebab.utils import save_json


def run(args, logger, json_files, output_suffix=""):
    all_counts = []
    all_generated_properties = []
    all_target_properties = []
    all_prop_hallucination = []

    metrics_accumulator = MetricsAccumulator()

    embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    for file_name in json_files:
        with open(file_name, "r") as file:
            data = json.load(file)

        counts, generated_property_counts, target_property_counts, prop_hallucination = evaluate_metrics(
            data=data,
            output_json_already_processed=True,
            logger=logger,
        )

        ground_truth = data["target"]
        predictions = data["generated_output"]

        metrics_for_update = compute_bipartite_metrics(
            ground_truth, predictions, normalize=True, embed_model=embed_model, matching_threshold=args.threshold
        )
        metrics_accumulator.update(metrics_for_update)

        all_counts.append(counts)
        all_prop_hallucination.append(prop_hallucination)
        all_generated_properties.append(generated_property_counts)
        all_target_properties.append(target_property_counts)

        bipartite_metrics_precision, bipartite_metrics_recall = compute_scores_across_documents(metrics_accumulator)

    unmatched_pair_fraction, extra_pred_entities_fraction, extra_gt_entities_fraction = compute_umatched_statistics(
        metrics_accumulator
    )

    bipartite_metrics_precision = dict(
        sorted(bipartite_metrics_precision.items(), key=lambda item: -item[1] if item[1] else 0)
    )
    bipartite_metrics_recall = dict(
        sorted(bipartite_metrics_recall.items(), key=lambda item: -item[1] if item[1] else 0)
    )

    total_counts = get_counts(all_counts)
    all_generated_properties = get_counts(all_generated_properties)
    all_target_properties = get_counts(all_target_properties)

    all_generated_properties = dict(sorted(all_generated_properties.items(), key=lambda item: item[1], reverse=True))
    all_target_properties = dict(sorted(all_target_properties.items(), key=lambda item: item[1], reverse=True))
    hallucination_fraction = get_hallucination_fraction(all_prop_hallucination)

    all_hallucination_fraction = dict(sorted(hallucination_fraction.items(), key=lambda item: item[0]))

    save_json(total_counts, args.output_dir / f"counts{output_suffix}.json", default_type=int)
    save_json(bipartite_metrics_precision, args.output_dir / f"bipartite_metrics_precision{output_suffix}.json")
    save_json(bipartite_metrics_recall, args.output_dir / f"bipartite_metrics_recall{output_suffix}.json")
    save_json(all_generated_properties, args.output_dir / f"generated_properties{output_suffix}.json", default_type=int)
    save_json(all_target_properties, args.output_dir / f"target_properties{output_suffix}.json", default_type=int)
    save_json(
        all_hallucination_fraction, args.output_dir / f"hallucination_fraction{output_suffix}.json", default_type=int
    )
    save_json(
        {
            "unmatched_pair_fraction": unmatched_pair_fraction,
            "unmatched_pair_count": metrics_accumulator.unmatched_pair_count,
            "unmatched_extra_entities_gt": metrics_accumulator.unmatched_extra_gt_count,
            "unmatched_extra_entities_pred": metrics_accumulator.unmatched_extra_pred_count,
            "unmatched_extra_entities_gt_fraction": extra_pred_entities_fraction,
            "unmatched_extra_entities_pred_fraction": extra_gt_entities_fraction,
        },
        args.output_dir / f"unmatched_fractions{output_suffix}.json",
    )
