# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd
import xlsxwriter

from kebab.contracts.entity import Entity, PropertySchema
from kebab.tasks.metrics.extraction.aesop.metric_helpers import (
    MetricsAccumulator,
    compute_bipartite_metrics,
    match_entities,
)
from kebab.tasks.metrics.extraction.aesop.metrics_factory import MetricsFactory
from kebab.tasks.metrics.extraction.calculator import ExtractionOutput, MetricCalculator


def document_debug_output_to_excel(
    debug_info: pd.DataFrame, evaluated_properties: list[str], output_path: Path, merge_rows: bool = True
) -> None:
    """Convert debug information to an Excel file.

    Args:
        debug_info: DataFrame containing debug information with entity and property data.
        evaluated_properties: List of property names to include in metrics calculations.
        output_path: Path where the Excel file should be saved.
        merge_rows: Whether to merge rows with same document_id for better readability.
    """
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

    if "entity_match_distance" not in debug_info.columns:
        debug_info["entity_match_distance"] = ""

    debug_info = pd.DataFrame(debug_info[ordered_columns])
    column_to_letter = {col: chr(65 + idx) for idx, col in enumerate(ordered_columns)}

    def col(column_name: str, start: int = 2, end: int = num_rows + 1) -> str:
        """Get column letter for a given column name."""
        letter = column_to_letter[column_name]
        return f"${letter}${start}:${letter}${end}"

    def conditions(property_id: str) -> str:
        return f"""{col("gt_entity_id")}, {{"<>{UNDEFINED_ID}", "?*"}}, {col("pred_entity_id")}, {{"<>{UNDEFINED_ID}", "?*"}}, {col("property_id")}, "{property_id}" """

    def property_precision_formula(property_id: str) -> str:
        """Return excel formula for property precision."""
        return f"""=SUMIFS({col("matched_scores")}, {conditions(property_id)}) / SUMIFS({col("num_pred_values")}, {conditions(property_id)}, {col("num_pred_values")}, ">0")"""

    def property_recall_formula(property_id: str) -> str:
        """Return excel formula for property recall."""
        return f"""=SUMIFS({col("matched_scores")}, {conditions(property_id)}) / SUMIFS({col("num_gt_values")}, {conditions(property_id)})"""

    debug_info.to_excel(writer, sheet_name=sheet_name, columns=ordered_columns, index=False)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = cast(xlsxwriter.Workbook, writer.book)
    worksheet = writer.sheets[sheet_name]
    if merge_rows:
        # Merge cells for document_id, original_text, and original_extraction
        merge_format = workbook.add_format({"align": "left", "valign": "top", "text_wrap": True})
        first_row_idx = 2
        for group_count in debug_info.groupby("document_id").size().to_numpy():
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
            first_row_idx += group_count

    metrics_start_col = chr(65 + len(ordered_columns) + 2)
    precision_col = chr(65 + len(ordered_columns) + 3)
    recall_col = chr(65 + len(ordered_columns) + 4)
    worksheet.write_row(f"{metrics_start_col}1", ["property_id", "property_precision", "property_recall"])
    for idx, property_id in enumerate(evaluated_properties):
        worksheet.write(f"{metrics_start_col}{idx + 2}", property_id)
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
    first_col = "D" if merge_rows else "A"
    worksheet.conditional_format(
        f"{first_col}2:{last_col}{num_rows + 1}", {"type": "formula", "criteria": "=ISEVEN(ROW())", "format": green_row}
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
            f"Ground truth has {self.num_gt_entities} entities in {self.num_documents} documents ({self.num_gt_entities / self.num_documents:.2f} entities per document)"
        )
        logger.info(
            f"After filtering, {self.num_gt_entities_after_filter} ground truth entities remain ({self.num_gt_entities_after_filter / self.num_documents:.2f} entities per document)"
        )
        logger.info(
            f"Prediction has {self.num_predicted_entities} entities ({self.num_predicted_entities / self.num_documents:.2f} entities per document)"
        )
        logger.info(
            f"After filtering, {self.num_predicted_entities_after_filter} predicted entities remain ({self.num_predicted_entities_after_filter / self.num_documents:.2f} entities per document)"
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

    def __init__(
        self,
        config: dict[str, Any],
        property_schema: PropertySchema,
        logger: logging.Logger | None = None,
        debug_output_path: Path | None = None,
    ):
        """Configure value-averaged-AESOP metric calculator.

        Args:
            config: deserialized from JSON configuration for AESOP metric.
            property_schema: Property schema for the entities.
            logger: Optional logger instance for logging metric computation progress.
            debug_output_path: Optional path to save debug information Excel files.
        """
        super().__init__(config, property_schema, logger, debug_output_path)
        self.metrics_factory = MetricsFactory(config, property_schema)
        self.logger = self.logger or logging.getLogger("Value-Averaged-AESOP")

    def preprocess_entities(self, entities: list[Entity]) -> list[Entity]:
        """Preprocess entities by removing specified properties.

        Args:
            entities: Iterable of Entity objects to preprocess.

        Returns:
            List of preprocessed Entity objects.
        """
        preprocessed_entities = entities.copy()
        for property_to_skip in self.metrics_factory.properties_to_skip:
            for entity in preprocessed_entities:
                if property_to_skip in entity.properties:
                    del entity.properties[property_to_skip]

        def has_complex_type(value: Any) -> bool:  # noqa: ANN401
            """Check if a value is of complex type (list or dict)."""
            return isinstance(value, list | dict)

        for entity in preprocessed_entities:
            for property_name, property_values in entity.properties.items():
                if any(has_complex_type(value) for value in property_values):
                    entity.properties[property_name] = [str(value) for value in property_values]
        return preprocessed_entities

    def run(self, prediction: Iterable[ExtractionOutput], ground_truth: Iterable[ExtractionOutput]) -> dict:
        """Calculate AESOP-based metrics."""
        metrics = {"per_document_metrics": {}, "dataset_metrics": {}}
        debug_info = {}
        metrics_accumulator = MetricsAccumulator()
        self.logger.info("Starting AESOP metric calculation")
        self.logger.info(f"Debug output directory: {self.debug_output_path}")
        if self.debug_output_path:
            self.debug_output_path.mkdir(parents=True, exist_ok=True)
        self.logger.info("Debug output directory created")

        def process_document(
            input_: tuple[int, tuple[ExtractionOutput, ExtractionOutput]],
        ) -> tuple[MetricsAccumulator, str, pd.DataFrame | None, Statistics]:
            """Compute metrics for a single document."""
            idx, (pred, gt) = input_
            self.logger.info(f"Processing document {idx + 1} with ID {gt.document.document_id}")

            gt_entities = self.preprocess_entities(gt.entities)
            pred_entities = self.preprocess_entities(pred.entities)

            context = {}

            context["gt_entities"] = gt_entities
            context["pred_entities"] = pred_entities

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

            self.logger.info(f"Document {idx + 1}: Matching entities")
            matching_info = match_entities(self.metrics_factory, **context, threshold=self.config["matching_threshold"])
            context["matching_info"] = matching_info
            self.logger.info(f"Document {idx + 1}: Computing metrics")
            metrics, document_debug_info = compute_bipartite_metrics(
                self.metrics_factory, **context, logger=self.logger
            )
            self.logger.info(f"Document {idx + 1}: Done")

            if document_debug_info is not None:
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
            if self.debug_output_path is not None:
                if document_debug_info is None:
                    self.logger.warning(f"No debug info for document {doc_id}, skipping")
                    continue
                self.logger.info(f"Saving debug info for document {doc_id}")
                document_debug_output_to_excel(
                    document_debug_info,
                    document_metrics["property_precision"].keys(),
                    self.debug_output_path / f"{doc_id}_debug_info.xlsx",
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
        if self.debug_output_path is not None:
            if len(debug_info) == 0:
                self.logger.warning("No debug info collected, skipping saving to XLSX")
            else:
                self.logger.info("Saving debug info to XLSX")
                debug_info_df = pd.concat(debug_info.values(), ignore_index=True)
                document_debug_output_to_excel(
                    debug_info_df,
                    metrics["dataset_metrics"]["property_precision"].keys(),
                    self.debug_output_path / "debug_info.xlsx",
                    merge_rows=False,
                )
                self.logger.info(f"Debug info saved to {self.debug_output_path / 'debug_info.csv'}")
        return metrics
