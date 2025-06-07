# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sentence_transformers import SentenceTransformer

from kebab.contracts.entity import Entity, Property, PropertySchema
from kebab.tasks.metrics.extraction.aesop import distances
from kebab.tasks.metrics.extraction.aesop.distances import (
    BinaryMatchDistance,
    EditDistance,
    ElementDistance,
    EmbeddingDistance,
    EntityDistance,
    PropertyScore,
    SetPropertyDistance,
    SingleValuePropertyDistance,
    TokenDistance,
    ValueMatchingRecordWihScores,
)
from kebab.tasks.metrics.extraction.aesop.metric_helpers import EntityMatcher, MetricsAccumulator, MetricsComputer
from kebab.tasks.metrics.extraction.calculator import ExtractionOutput, MetricCalculator, MetricConfig


@dataclass
class ValueAveragedAesopConfig(MetricConfig):
    """Value-averaged-AESOP metric configuration."""

    matching_score_function: Callable[[Entity, Entity], float]
    matching_threshold: float
    property_score_functions: dict[str, Callable[[Entity, Entity, Property], ValueMatchingRecordWihScores]]
    property_schema: PropertySchema
    debug_output_dir: Path | None = None

    @staticmethod
    def from_dict(config: dict[str, Any], property_schema: PropertySchema) -> ValueAveragedAesopConfig:
        """Create value-averaged-AESOP metric configuration from dictionary."""

        def get_element_distance(element_distance_params: dict[str, Any]) -> ElementDistance:
            """Get element distance function from parameters."""
            element_distance_cls_name = element_distance_params["name"]
            return getattr(distances, element_distance_cls_name).from_dict(element_distance_params.get("params", {}))

        default_property_score = PropertyScore(
            SetPropertyDistance(get_element_distance(config["default_property_distance"]))
        )

        property_score_functions = defaultdict(lambda: default_property_score)
        for property_name, score_config in config["property_distance_functions"].items():
            if property_name not in property_schema.properties:
                raise ValueError(f"Property '{property_name}' is not present in the property schema.")
            element_distance = get_element_distance(score_config)
            property_score_functions[property_name] = PropertyScore(
                SetPropertyDistance(element_distance)
                if property_schema.properties[property_name].is_collection
                else SingleValuePropertyDistance(element_distance)
            )
        debug_output_dir = config.get("debug_output_dir")
        return ValueAveragedAesopConfig(
            matching_score_function=EntityDistance.from_dict(config["entity_distance"], property_schema),
            matching_threshold=config["matching_threshold"],
            property_score_functions=property_score_functions,
            property_schema=property_schema,
            debug_output_dir=Path(debug_output_dir) if debug_output_dir else None,
        )


def document_debug_output_to_excel(
    debug_info: pd.DataFrame, evaluated_properties: list[str], output_path: Path, merge_rows: bool = True
) -> None:
    """Convert debug information to an Excel file."""
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")

    # Convert the dataframe to an XlsxWriter Excel object. Note that we turn off
    # the default header and skip one row to allow us to insert a user defined
    # header.
    sheet_name = "debug_info"
    num_rows = debug_info.shape[0]
    ordered_columns = [
        "document_id",
        "original_text",
        "gt_entity_id",
        "pred_entity_id",
        "original_pred_property_id",
        "property_id",
        "gt_values",
        "pred_values",
        "matched_scores",
        "num_gt_values",
        "num_pred_values",
        "entity_match_distance",
    ]
    debug_info = debug_info[ordered_columns]  # type: ignore
    column_to_letter = {col: chr(65 + idx) for idx, col in enumerate(ordered_columns)}
    print(f"Column to letter mapping: {column_to_letter}")

    def col(column_name: str, start: int = 2, end: int = num_rows + 1) -> str:
        """Get column letter for a given column name."""
        letter = column_to_letter[column_name]
        return f"${letter}${start}:${letter}${end}"

    def conditions(property_id: str) -> str:
        return f"""{col("gt_entity_id")}, "?*", {col("pred_entity_id")}, "?*", {col("property_id")}, "{property_id}" """

    def property_precision_formula(property_id: str) -> str:
        """Return excel formula for property precision."""
        return f"""=SUMIFS({col("matched_scores")}, {conditions(property_id)}) / SUMIFS({col("num_pred_values")}, {conditions(property_id)})"""

    def property_recall_formula(property_id: str) -> str:
        """Return excel formula for property recall."""
        return f"""=SUMIFS({col("matched_scores")}, {conditions(property_id)}) / SUMIFS({col("num_gt_values")}, {conditions(property_id)})"""

    debug_info.to_excel(writer, sheet_name=sheet_name, columns=ordered_columns, index=False)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    if merge_rows:
        # Merge cells for document_id, original_text, and original_extraction
        merge_format = workbook.add_format({"align": "left", "valign": "top", "text_wrap": True})
        first_row_idx = 2
        for group_count in debug_info.groupby("document_id").size().to_numpy():
            print(f"Processing group with count: {group_count}")
            worksheet.merge_range(
                f"{col('document_id', start=first_row_idx, end=first_row_idx + group_count - 1)}",
                debug_info["document_id"].to_numpy()[first_row_idx],
                merge_format,
            )
            worksheet.merge_range(
                f"{col('original_text', start=first_row_idx, end=first_row_idx + group_count - 1)}",
                debug_info["original_text"].to_numpy()[first_row_idx],
                merge_format,
            )
            # worksheet.merge_range(
            #     f"{col('original_extraction', start=first_row_idx, end=first_row_idx + group_count - 1)}",
            #     debug_info["original_extraction"].to_numpy()[first_row_idx],
            #     merge_format,
            # )
            first_row_idx += group_count

    metrics_start_col = chr(65 + len(ordered_columns) + 2)
    precision_col = chr(65 + len(ordered_columns) + 3)
    recall_col = chr(65 + len(ordered_columns) + 4)
    worksheet.write_row(f"{metrics_start_col}1", ["property_id", "property_precision", "property_recall"])
    for idx, property_id in enumerate(evaluated_properties):
        worksheet.write(f"{metrics_start_col}{idx + 2}", property_id)
        print(property_precision_formula(property_id))
        print(property_recall_formula(property_id))
        worksheet.write_formula(f"{precision_col}{idx + 2}", property_precision_formula(property_id))
        worksheet.write_formula(f"{recall_col}{idx + 2}", property_recall_formula(property_id))
    worksheet.write(f"{metrics_start_col}{len(evaluated_properties) + 2}", "Average")
    worksheet.write_formula(
        precision_col + str(len(evaluated_properties) + 2),
        f"""AVERAGEIF({precision_col}2:{precision_col}{len(evaluated_properties) + 1}, ">=0")""",
    )
    worksheet.write_formula(
        recall_col + str(len(evaluated_properties) + 2),
        f"""AVERAGEIF({recall_col}2:{recall_col}{len(evaluated_properties) + 1}, ">=0")""",
    )
    worksheet.freeze_panes(1, 2)  # Freeze the first row and first two columns
    green_row = workbook.add_format({"bg_color": "#dbf2d2"})
    header_format = workbook.add_format({"bold": True})
    worksheet.set_row(0, None, header_format)  # Set header format
    last_col = column_to_letter[ordered_columns[-1]]
    if merge_rows:
        worksheet.conditional_format(
            f"D2:{last_col}{num_rows + 1}", {"type": "formula", "criteria": "=ISEVEN(ROW())", "format": green_row}
        )
    else:
        worksheet.conditional_format(
            f"A2:{last_col}{num_rows + 1}", {"type": "formula", "criteria": "=ISEVEN(ROW())", "format": green_row}
        )
    worksheet.autofit(300)  # Adjust column widths to fit content
    writer.close()  # Save the Excel file


@dataclass
class Statistics:
    """Entity and property statistics for extraction task."""

    num_gt_entities: int = 0
    num_gt_entities_after_filter: int = 0
    num_predicted_entities: int = 0
    num_predicted_entities_after_filter: int = 0
    num_documents: int = 0
    gt_property_counts: Counter[str] = field(default_factory=Counter)
    pred_property_counts: Counter[str] = field(default_factory=Counter)

    def update(self, other: Statistics) -> None:
        """Update statistics with another EntityStatistics object."""
        self.num_gt_entities += other.num_gt_entities
        self.num_gt_entities_after_filter += other.num_gt_entities_after_filter
        self.num_predicted_entities += other.num_predicted_entities
        self.num_documents += other.num_documents
        self.pred_property_counts.update(other.pred_property_counts)
        self.gt_property_counts.update(other.gt_property_counts)

    def log_statistics(self, logger: logging.Logger) -> None:
        """Log the entity statistics."""
        logger.info("Entity statistics:")
        logger.info(
            f"Ground truth has {self.num_gt_entities} in {self.num_documents} documents ({self.num_gt_entities / self.num_documents:.2f} entities per document)"
        )
        logger.info(
            f"After filtering, {self.num_gt_entities_after_filter} target entities remain ({self.num_gt_entities_after_filter / self.num_documents:.2f} entities per document)"
        )
        logger.info(
            f"Prediction has {self.num_predicted_entities} entities ({self.num_predicted_entities / self.num_documents:.2f} entities per document)"
        )
        logger.info("Property statistics:")
        for key, value in self.gt_property_counts.items():
            if key in self.pred_property_counts:
                logger.info(f"{key}: {value} in ground truth, {self.pred_property_counts[key]} predicted")
            else:
                logger.info(f"{key}: {value} in ground truth, 0 predicted")
        for key, value in self.pred_property_counts.items():
            if key not in self.gt_property_counts:
                logger.info(f"{key}: 0 in ground truth, {value} predicted")


class ValueAveragedAesopMetricCalculator(MetricCalculator):
    """Value-averaged-AESOP metric calculator.
    Computes AESOP metric as described in the paper: https://arxiv.org/pdf/2402.04437
    with the following modification:
    We don't average property scores on entity level to obtain entity similarities.
    Instead, we compute similarity scores for each pair of matched values of that property in the dataset.
    Then, we average these scores to obtain precision for the property.
    Finally, we average precision scores for all properties to obtain the average precision.
    Similarly, we compute recall scores for each property and average them to obtain the average recall.
    """

    def __init__(self, config: ValueAveragedAesopConfig, logger: logging.Logger | None = None):
        """Configure value-averaged-AESOP metric calculator."""
        self.config = config
        self.logger = logger or logging.getLogger("Value-Averaged-AESOP")

    def run(self, prediction: Iterable[ExtractionOutput], ground_truth: Iterable[ExtractionOutput]) -> dict:
        """Calculate AESOP-based metrics."""
        metrics = {"per_document_metrics": {}, "dataset_metrics": {}}
        debug_info = {}
        metrics_accumulator = MetricsAccumulator()
        self.logger.info("Starting AESOP metric calculation")
        self.logger.info(f"Debug output directory: {self.config.debug_output_dir}")
        if self.config.debug_output_dir:
            self.config.debug_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Debug output directory created")

        def process_document(
            input_: tuple[int, tuple[ExtractionOutput, ExtractionOutput]],
        ) -> tuple[MetricsAccumulator, str, pd.DataFrame, Statistics]:
            """Compute metrics for a single document."""
            idx, (pred, gt) = input_
            self.logger.info(f"Processing document {idx + 1}")

            gt_entities = [
                entity
                for entity in gt.entities
                if not entity.properties.get("type") or "time" not in entity.properties["type"]
            ]
            pred_entities = [
                entity
                for entity in pred.entities
                if not entity.properties.get("type") or "time" not in entity.properties["type"]
            ]
            gt_property_counts = Counter()
            for entity in gt_entities:
                gt_property_counts.update(entity.properties.keys())
            pred_property_counts = Counter()
            for entity in pred_entities:
                pred_property_counts.update(entity.properties.keys())
            stats = Statistics(
                num_gt_entities=len(gt.entities),
                num_predicted_entities=len(pred.entities),
                num_gt_entities_after_filter=len(gt_entities),
                num_predicted_entities_after_filter=len(pred_entities),
                num_documents=1,
                gt_property_counts=gt_property_counts,
                pred_property_counts=pred_property_counts,
            )
            entity_matcher = EntityMatcher(gt_entities, pred_entities)
            self.logger.info(f"Document {idx + 1}: Matching entities")
            matched_pairs = entity_matcher.match(self.config.matching_score_function, self.config.matching_threshold)
            self.logger.info(f"Document {idx + 1}: Computing metrics")
            metrics_computer = MetricsComputer(gt_entities, pred_entities, self.config.property_schema)
            metrics, document_debug_info = metrics_computer.compute_bipartite_metrics(
                matched_pairs, self.config.property_score_functions
            )
            self.logger.info(f"Document {idx + 1}: Done")
            if not document_debug_info.empty:
                document_debug_info["document_id"] = gt.document.document_id
                document_debug_info["original_text"] = gt.document.data.get("text", "")
            return metrics, gt.document.document_id, document_debug_info, stats

        entity_and_property_stats = Statistics()
        start = time.time()
        for idx, (document_metrics_accumulator, doc_id, document_debug_info, document_stats) in enumerate(
            map(
                process_document,
                enumerate(zip(prediction, ground_truth, strict=True)),
            )
        ):
            entity_and_property_stats.update(document_stats)
            document_metrics = document_metrics_accumulator.accumulate_metrics()
            metrics["per_document_metrics"][doc_id] = document_metrics
            metrics_accumulator.update(document_metrics_accumulator)
            if self.config.debug_output_dir:
                document_debug_info.to_csv(self.config.debug_output_dir / f"{doc_id}_debug_info.csv", index=False)
                document_debug_output_to_excel(
                    document_debug_info,
                    document_metrics["property_precision"].keys(),
                    self.config.debug_output_dir / f"{doc_id}_debug_info.xlsx",
                    merge_rows=False,
                )
                debug_info[doc_id] = document_debug_info
            elapsed = time.time() - start
            self.logger.info(f"Time elapsed: {elapsed:.2f} seconds")
            self.logger.info(f"Average processing time: {elapsed / (idx + 1):.2f} seconds per document")
        self.logger.info("Accumulating scores across documents")
        elapsed = time.time() - start
        self.logger.info(f"Total time elapsed: {elapsed:.2f} seconds")
        start = time.time()
        metrics["dataset_metrics"] = metrics_accumulator.accumulate_metrics()
        elapsed = time.time() - start
        self.logger.info(f"Metrics accumulation time: {elapsed:.2f} seconds")
        entity_and_property_stats.log_statistics(self.logger)
        if self.config.debug_output_dir:
            debug_info_df = pd.concat(debug_info.values(), ignore_index=True)
            debug_info_df.to_csv(self.config.debug_output_dir / "all_debug_info.csv", index=False)
            document_debug_output_to_excel(
                debug_info_df,
                metrics["dataset_metrics"]["property_precision"].keys(),
                self.config.debug_output_dir / "debug_info.xlsx",
                merge_rows=False,
            )
            self.logger.info(f"Debug info saved to {self.config.debug_output_dir / 'debug_info.csv'}")
        return metrics


def make_default_value_averaged_aesop_config(
    property_schema: PropertySchema, matching_threshold: float = 1.0, embed_model: SentenceTransformer | None = None
) -> ValueAveragedAesopConfig:
    """Create default value-averaged-AESOP metric config."""
    # Set score functions are used for properties for which we expect multiple values.
    # Score function always returns a list of scores, even if there is only one value.
    if embed_model is None:
        embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # type: ignore

    set_embedding_score = PropertyScore(SetPropertyDistance(EmbeddingDistance()))
    set_token_score = PropertyScore(SetPropertyDistance(TokenDistance()))
    str_score = PropertyScore(SingleValuePropertyDistance(BinaryMatchDistance()))
    edit_score = PropertyScore(SetPropertyDistance(EditDistance()))
    embedding_score = PropertyScore(SingleValuePropertyDistance(EmbeddingDistance(model=embed_model)))

    property_to_score = defaultdict(lambda: set_token_score)
    property_to_score.update(
        {
            "descriptions": set_embedding_score,
            "definitions": set_embedding_score,
            "objectives": set_embedding_score,
            "risks": set_embedding_score,
            "purpose": embedding_score,
            "url": str_score,
            "email": str_score,
            "password": str_score,
            "name": edit_score,
        }
    )

    return ValueAveragedAesopConfig(
        matching_score_function=EntityDistance(
            property_schema,
            {"name": (SetPropertyDistance(TokenDistance()), 1)},
            default_property_distance=SetPropertyDistance(TokenDistance()),
            default_property_weight=0,
        ),
        matching_threshold=matching_threshold,
        property_score_functions=property_to_score,
        property_schema=property_schema,
        debug_output_dir=None,
    )
