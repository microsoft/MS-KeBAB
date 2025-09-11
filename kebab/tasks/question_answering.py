# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import statistics
from abc import abstractmethod
from collections.abc import Iterable
from logging import Logger
from pathlib import Path

import evaluate

from kebab.contracts.document import Document
from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.utils.io_helpers import (
    DocumentJsonlReader,
    EntityJsonlReader,
    StringLineReader,
    StringLineWriter,
    resolve_path,
    save_dict_to_json,
)


class QuestionAnsweringTaskBase(Task):
    """Base class for question-answering benchmark task."""

    __questions: Path
    __ground_truth: Path

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Return task type."""

    def __init__(
        self,
        name: str,
        questions: Path,
        ground_truth: Path,
        schema: Path | None,
        data: Path | None = None,
    ):
        """Initialize a question-answering task."""
        super().__init__(name, schema=schema, data=data)
        self.__questions = resolve_path(questions)
        self.__ground_truth = resolve_path(ground_truth)

    def read_items(self) -> Iterable[str]:
        """
        Read string questions from file.

        Returns:
            Iterable[str]: An iterable of string objects.
        """
        return StringLineReader(self.__questions).read_items()

    def write_items(self, path: Path, items: Iterable[str]) -> None:
        """
        Write predicted answers to the specified path.

        Args:
            path: The file path where the predicted answers should be written.
            items: An iterable of string answers
        """
        StringLineWriter(path).write_items([answer.replace("\n", " ") for answer in items])

    def evaluate(
        self,
        predictions: Path,
        result_output_path: Path | None = None,
        logger: Logger | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the question-answering task."""
        if logger:
            logger.info("Starting evaluation for the question-answering task.")

        predicted_vals = StringLineReader(predictions).read_items()
        targets = StringLineReader(self.__ground_truth).read_items()
        predictions_and_targets = zip(predicted_vals, targets, strict=True)
        accuracy = [1 if item[0] == item[1] else 0 for item in predictions_and_targets]

        metrics = {}
        metrics["exact_match_accuracy"] = statistics.mean(accuracy)

        rouge = evaluate.load("rouge")
        rouge_metrics = rouge.compute(predictions=predicted_vals, references=targets)
        if rouge_metrics is not None:
            metrics |= rouge_metrics

        bertscore = evaluate.load("bertscore")
        model_type = "microsoft/deberta-xlarge-mnli"
        bertscore_metrics = bertscore.compute(
            predictions=predicted_vals, references=targets, lang="en", model_type=model_type
        )
        if bertscore_metrics is not None:
            metrics["bertscore_model"] = model_type
            metrics |= {"bertscore_" + metric: value for metric, value in bertscore_metrics.items()}

        if logger:
            logger.info("Evaluation metrics calculated successfully.")
            logger.info(f"Metrics: {metrics}")
        if result_output_path:
            save_dict_to_json(metrics, result_output_path)

        return metrics


class QuestionAnsweringUsingDocumentsTask(QuestionAnsweringTaskBase):
    """Represents an end-to-end question-answering benchmark task with its data files."""

    __documents: Path

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.QuestionAnsweringUsingDocuments

    def __init__(
        self,
        name: str,
        documents: Path,
        questions: Path,
        ground_truth: Path,
        data: Path | None = None,
    ):
        """Initialize an end-to-end question-answering completion task."""
        super().__init__(
            name,
            questions=questions,
            ground_truth=ground_truth,
            schema=None,
            data=data,
        )  # kb schema is not used in this task
        self.__documents = Path(documents)

    @property
    def documents(self) -> Path:
        """Return path to documents."""
        return self.__documents

    def read_documents(self) -> Iterable[Document]:
        """
        Read data items from a JSONL file containing documents.

        Returns:
            Iterable[Document]: An iterable of `Document` objects.
        """
        return DocumentJsonlReader(self.__documents).read_items()


class QuestionAnsweringUsingKBTask(QuestionAnsweringTaskBase):
    """Represents a kb-augmented question-answering benchmark task with its data files."""

    __kb: Path

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.QuestionAnsweringUsingKB

    @property
    def kb(self) -> Path:
        """Return path to knowledge base."""
        return self.__kb

    def __init__(
        self,
        name: str,
        kb: Path,
        questions: Path,
        ground_truth: Path,
        schema: Path | None = None,
        data: Path | None = None,
    ):
        """Initialize a kb-augmented question-answering task."""
        super().__init__(name, questions=questions, ground_truth=ground_truth, schema=schema, data=data)
        self.__kb = Path(kb)

    def read_kb(self) -> Iterable[Entity]:
        """
        Read entities from knowledge base.

        Returns:
            Iterable[Entity]: An iterable of `Entity` objects.
        """
        return EntityJsonlReader(self.__kb).read_items()

    def remove_sources_from_kb(self, kb: list[Entity], remove_sources_list: list[str]) -> list[Entity]:
        """Return a new KB after removing property values corresponding to specified sources from all entities.

        Args:
            kb: A list of entities
            remove_sources_list: A list of sources IDs to remove.

        Returns:
            A new list of entities with filtered values.
        """
        updated_kb = [entity.remove_sources(remove_sources_list) for entity in kb]
        updated_kb = [entity for entity in updated_kb if entity is not None]  # remove any empty entities
        return updated_kb
