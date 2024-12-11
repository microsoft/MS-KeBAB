# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Callable, Iterable

from kebab.contracts.document import Document
from kebab.contracts.entity import Entity
from kebab.contracts.task import Task, TaskInstance
from kebab.utils.io_helpers import DocumentJsonlReader, EntityListJsonlReader, EntityListJsonlWriter, save_dict_to_json


class ExtractionTaskInstance(TaskInstance):
    """Represents an extraction benchmark task instance with its data files."""

    __data_extracts: Path
    __data_ground_truth_extracted_entities: Path | None
    __read_items_fun: Callable[[], Iterable[tuple[Document, list[Entity] | None]]]
    __write_items_fun: Callable[[Path, Iterable[list[Entity]]], None]

    @property
    def data_extracts(self) -> Path:
        """Return path to extracts."""
        return self.__data_extracts

    @property
    def data_ground_truth_extracted_entities(self) -> Path | None:
        """Return path to ground truth extracted entities."""
        return self.__data_ground_truth_extracted_entities

    def __init__(
        self,
        name: str,
        task: Task,
        extracts: str,
        ground_truth_extracted_entities: str | None = None,
        read_items_fun: Callable[[], Iterable[tuple[Document, list[Entity] | None]]] | None = None,
        write_items_fun: Callable[[Path, Iterable[list[Entity]]], None] | None = None,
    ):
        """Initialize an extraction task instance."""
        super().__init__(name, task)
        self.__data_extracts = Path(extracts)
        if ground_truth_extracted_entities is not None:
            self.__data_ground_truth_extracted_entities = Path(ground_truth_extracted_entities)
        self.__read_items_fun = read_items_fun if read_items_fun is not None else self.__default_read_items
        self.__write_items_fun = (
            write_items_fun if write_items_fun is not None else ExtractionTaskInstance.__default_write_items
        )

    def __default_read_items(self) -> Iterable[tuple[Document, list[Entity] | None]]:
        extracts = DocumentJsonlReader(self.data_extracts).read_items()
        entity_lists = (
            EntityListJsonlReader(self.data_ground_truth_extracted_entities).read_items()
            if self.data_ground_truth_extracted_entities is not None
            else iter([])
        )
        return zip_longest(extracts, entity_lists)

    def read_items(self) -> Iterable[tuple[Document, list[Entity] | None]]:
        """
        Read data items, with optional ground-truth extracted entities.

        Returns:
        Iterable[Tuple[Document, List[Entity] | None]]: An iterable of tuples, each containing a
        `Document` and an optional list of `Entity` objects.
        """
        return self.__read_items_fun()

    def write_items(self, path: Path, items: Iterable[list[Entity]]) -> None:
        """
        Write output items, extracted entities, to the specified path.

        Args:
        path: The file path where the items should be written.
        items: An iterable of lists of extracted `Entity` objects to be written to the file.
        """
        self.__write_items_fun(path, items)

    def evaluate(
        self,
        output_to_evaluate: Path,  # noqa: ARG002
        eval_result_path: Path | None = None,
    ) -> dict[str, float]:
        """Evaluate an output for the extraction task instance."""
        if hasattr(self, "__data_ground_truth_extracted_entities"):
            raise ValueError("Can not evaluate on heldout Extraction task instance")

        # TODO(bmitra): Implement actual metric computation
        eval_result = {"primary_extraction_metric": 0.8, "secondary_extraction_metric": 0.6}

        if eval_result_path:
            save_dict_to_json(eval_result, eval_result_path)

        return eval_result

    @staticmethod
    def __default_write_items(path: Path, items: Iterable[list[Entity]]) -> None:
        EntityListJsonlWriter(path).write_items(items)
