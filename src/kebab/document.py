# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Classes and utilities for working with documents."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Self


@dataclass
class DocumentFieldSchema:
    """A schema for a field in a document."""

    field_id: str
    description: str
    display_name: str

    def to_dict(self) -> dict[str, str]:
        """Convert the field schema to a dictionary."""
        return {
            "field_id": self.field_id,
            "description": self.description,
            "display_name": self.display_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Self:
        """Create a field schema from a dictionary."""
        return cls(
            field_id=data["field_id"],
            description=data["description"],
            display_name=data["display_name"] if "display_name" in data else data["field_id"],
        )


@dataclass
class DocumentSchema:
    """A schema for a document, specifying the types and descriptions of each document field."""

    schema_id: str
    fields: dict[str, DocumentFieldSchema]

    def to_dict(self) -> dict[str, Any]:
        """Convert the schema to a dictionary."""
        return {
            "schema_id": self.schema_id,
            "fields": [field.to_dict() for field in self.fields.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a schema from a dictionary."""
        fields = {
            field.field_id: field
            for field in (DocumentFieldSchema.from_dict(field_data) for field_data in data["fields"])
        }

        assert data["schema_id"], "Schema ID must be provided."
        assert len(fields) == len(data["fields"]), "Field IDs must be unique."

        return cls(
            schema_id=data["schema_id"],
            fields=fields,
        )


@dataclass
class Document:
    """Represents a document with its metadata and content."""

    document_id: str
    """Unique identifier for the document, needed to track provenance."""

    schema: DocumentSchema
    """Schema that defines the structure of the document."""

    data: dict[str, Any]
    """Dictionary that must be serializable to JSON."""

    def to_json(self) -> str:
        """Convert the document to a JSON string."""
        return json.dumps(
            {
                "document_id": self.document_id,
                "schema_id": self.schema.schema_id,
                "data": self.data,
            },
        )

    @classmethod
    def from_json(cls, json_str: str, schemas: dict[str, DocumentSchema]) -> Self:
        """Create a document from a JSON string."""
        data = json.loads(json_str)

        return cls(
            document_id=data["document_id"],
            schema=schemas[data["schema_id"]],
            data=data["data"],
        )


class DocumentUtilities:
    """Utility functions for working with collections of documents."""

    @classmethod
    def load_collection(
        cls,
        path_to_schema_dir: pathlib.Path,
        path_to_documents_dir: pathlib.Path,
    ) -> Iterable[Document]:
        """Load schemas and a collection of documents from a directory."""
        assert path_to_documents_dir.exists(), f"Directory not found: {path_to_documents_dir}"

        schemas = cls.load_schemas(path_to_schema_dir)

        for documents_path in path_to_documents_dir.glob("*.jsonl"):
            yield from cls.load_documents(documents_path, schemas)

    @classmethod
    def load_schemas(cls, path: pathlib.Path) -> dict[str, DocumentSchema]:
        """Load a collection of schemas from a directory."""
        assert path.exists(), f"Directory not found: {path}"

        schemas = {}
        for schema_path in path.glob("*.json"):
            with open(schema_path, encoding="utf-8") as file:
                schema_data = json.load(file)
                schema = DocumentSchema.from_dict(schema_data)
                schemas[schema.schema_id] = schema
        return schemas

    @classmethod
    def load_documents(cls, path: pathlib.Path, schemas: dict[str, DocumentSchema]) -> Iterable[Document]:
        """Load a list of documents from a JSON-lines file."""
        assert path.exists(), f"File not found: {path}"

        with open(path, encoding="utf-8") as f:
            for line in f:
                yield Document.from_json(line, schemas=schemas)

    @classmethod
    def save_documents(cls, documents: Iterable[Document], path: pathlib.Path) -> None:
        """Save a list of documents as a JSON-lines file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for document in documents:
                f.write(document.to_json() + "\n")
