# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import statistics
from abc import abstractmethod
from collections.abc import Iterable
from logging import Logger
from pathlib import Path

from kebab.contracts.document import Document
from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskType
from kebab.utils.io_helpers import (
    DocumentJsonlReader,
    EntityJsonlReader,
    StringLineReader,
    StringLineWriter,
    save_dict_to_json,
)


class QuestionAnsweringTaskBase(Task):
    """Base class for question-answering benchmark task."""

    __data_questions: Path
    __data_ground_truth_answers: Path

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Return task type."""

    def __init__(
        self,
        name: str,
        questions: str,
        ground_truth_answers: str,
        schema: str,
    ):
        """Initialize a question-answering task."""
        super().__init__(name, schema=schema)
        self.__data_questions = Path(questions)
        self.__data_ground_truth_answers = Path(ground_truth_answers)

    def read_items(self) -> Iterable[str]:
        """
        Read string questions from file.

        Returns:
            Iterable[str]: An iterable of string objects.
        """
        return StringLineReader(self.__data_questions).read_items()

    def write_items(self, path: Path, items: Iterable[str]) -> None:
        """
        Write predicted answers to the specified path.

        Args:
            path: The file path where the predicted answers should be written.
            items: An iterable of string answers
        """
        StringLineWriter(path).write_items([answer.replace("\n", "") for answer in items])

    def evaluate(
        self,
        output_to_evaluate: Path,
        eval_result_path: Path | None = None,
        logger: Logger | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the question-answering task."""
        if logger:
            logger.info("Starting evaluation for the question-answering task.")

        predictions = StringLineReader(output_to_evaluate).read_items()
        targets = StringLineReader(self.__data_ground_truth_answers).read_items()
        predictions_and_targets = zip(predictions, targets, strict=True)
        accuracy = [1 if item[0] == item[1] else 0 for item in predictions_and_targets]

        metrics = {}
        metrics["mean_accuracy"] = statistics.mean(accuracy)

        # TODO (bmitra): Consider adding more evaluation metrics, such as from KBLaM, to assess the
        # quality of predicted contents.

        if logger:
            logger.info("Evaluation metrics calculated successfully.")
            logger.info(f"Metrics: {metrics}")
        if eval_result_path:
            save_dict_to_json(metrics, eval_result_path)

        return metrics


class QuestionAnsweringE2ETask(QuestionAnsweringTaskBase):
    """Represents an end-to-end question-answering benchmark task with its data files."""

    __data_documents: Path

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.QuestionAnsweringE2E

    def __init__(
        self,
        name: str,
        documents: str,
        questions: str,
        ground_truth_answers: str,
    ):
        """Initialize an end-to-end question-answering completion task."""
        super().__init__(
            name, questions=questions, ground_truth_answers=ground_truth_answers, schema=""
        )  # kb schema is not used in this task
        self.__data_documents = Path(documents)

    @property
    def data_documents(self) -> Path:
        """Return path to documents."""
        return self.__data_documents

    def read_documents(self) -> Iterable[Document]:
        """
        Read data items from a JSONL file containing documents.

        Returns:
            Iterable[Document]: An iterable of `Document` objects.
        """
        return DocumentJsonlReader(self.__data_documents).read_items()


class QuestionAnsweringUsingKBTask(QuestionAnsweringTaskBase):
    """Represents a kb-augmented question-answering benchmark task with its data files."""

    __data_kb: Path

    @property
    def task_type(self) -> TaskType:
        """Return task type."""
        return TaskType.QuestionAnsweringUsingKB

    @property
    def data_kb(self) -> Path:
        """Return path to knowledge base."""
        return self.__data_kb

    def __init__(
        self,
        name: str,
        kb: str,
        questions: str,
        ground_truth_answers: str,
        schema: str,
    ):
        """Initialize a kb-augmented question-answering task."""
        super().__init__(name, questions=questions, ground_truth_answers=ground_truth_answers, schema=schema)
        self.__data_kb = Path(kb)

    def read_kb(self) -> Iterable[Entity]:
        """
        Read entities from knowledge base.

        Returns:
            Iterable[Entity]: An iterable of `Entity` objects.
        """
        return EntityJsonlReader(self.__data_kb).read_items()

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
