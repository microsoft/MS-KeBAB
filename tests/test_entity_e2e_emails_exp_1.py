# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the end-to-end workflow on the Emails Experiment 1."""

from pathlib import Path

from kebab.contracts.document import Document, DocumentSchema, DocumentUtilities
from kebab.contracts.entity import Entity, EntityUtilities, PropertySchema


def test_load_document_collection(document_schema_dir_path: Path, emails_exp_1_dir_path: Path):
    """Test loading a collection of documents from the emails experiment 1."""
    documents = list(
        DocumentUtilities.load_collection(
            path_to_schema_dir=document_schema_dir_path,
            path_to_documents_dir=emails_exp_1_dir_path / "documents",
        )
    )

    assert len(documents) == 1
    document = documents[0]

    assert document.document_id == "email_1"

    assert document.schema.schema_id == "email"

    assert document.data["from"] == "David <david@email.com>"
    assert document.data["to"] == ["John Doe <john@email.com>", "receipient2@email.com"]
    assert document.data["subject"] == "Hello"
    assert document.data["date"] == "2024-04-25T00:00:00Z"
    assert document.data["body"] == "Hello, John, Happy Birthday!"


def test_schema_to_from_dict(document_schema_dir_path: Path):
    """Test converting a schema to and from a dictionary."""
    schema = next(iter(DocumentUtilities.load_schemas(document_schema_dir_path).values()))

    schema_dict = schema.to_dict()
    new_schema = DocumentSchema.from_dict(schema_dict)

    assert schema == new_schema


def test_document_to_from_json(document_schema_dir_path: Path, emails_exp_1_dir_path: Path):
    """Test converting a document to and from a dictionary."""
    documents = list(
        DocumentUtilities.load_collection(
            path_to_schema_dir=document_schema_dir_path,
            path_to_documents_dir=emails_exp_1_dir_path / "documents",
        )
    )

    document = documents[0]

    json_str = document.to_json()
    new_document = Document.from_json(json_str, {document.schema.schema_id: document.schema})

    assert document == new_document


def test_load_property_schema(emails_exp_1_dir_path: Path):
    """Test loading a property schema from the emails experiment 1."""
    schema = EntityUtilities.load_schema(emails_exp_1_dir_path / "properties" / "property_schema.json")
    assert schema.name == "email_exp_schema"
    assert len(schema.data_types) == 1
    assert len(schema.properties) == 3

    assert schema.properties["name"].display_name == "name"
    assert schema.properties["name"].data_type == schema.data_types["text"]
    assert schema.properties["name"].is_collection is False

    assert schema.properties["email_addresses"].display_name == "email addresses"
    assert schema.properties["email_addresses"].data_type == schema.data_types["text"]
    assert schema.properties["email_addresses"].is_collection is True

    assert schema.properties["date_of_birth"].data_type == schema.data_types["text"]
    assert schema.properties["date_of_birth"].is_collection is False


def test_property_schema_to_from_json(emails_exp_1_dir_path: Path):
    """Test converting a property schema to and from a dictionary."""
    schema = EntityUtilities.load_schema(emails_exp_1_dir_path / "properties" / "property_schema.json")

    schema_dict = schema.to_dict()
    new_schema = PropertySchema.from_dict(schema_dict)

    assert len(schema.data_types) == len(new_schema.data_types)

    for data_type_id, data_type in schema.data_types.items():
        new_data_type = new_schema.data_types[data_type_id]
        assert data_type == new_data_type

    assert len(schema.properties) == len(new_schema.properties)

    for property_id, prop in schema.properties.items():
        new_property = new_schema.properties[property_id]
        assert prop == new_property

    assert schema == new_schema


def test_load_entities(emails_exp_1_dir_path: Path):
    """Test loading a collection of entities from the emails experiment 1."""
    schema = EntityUtilities.load_schema(emails_exp_1_dir_path / "properties" / "property_schema.json")
    entities = list(EntityUtilities.load_entities(emails_exp_1_dir_path / "extracted_entities"))
    assert len(entities) == 1

    entity = entities[0]

    name_property_id = [p for p in schema.properties.values() if p.display_name == "name"][0].property_id
    assert entity.properties[name_property_id] == ["John Doe"]

    emails_property_id = [p for p in schema.properties.values() if p.display_name == "email addresses"][0].property_id
    assert entity.properties[emails_property_id] == ["john@email.com"]


def test_entity_without_metadata(emails_exp_1_dir_path: Path):
    """Test metadata clearing."""
    entities = list(EntityUtilities.load_entities(emails_exp_1_dir_path / "extracted_entities"))
    assert len(entities) == 1

    entity = entities[0]
    entity.metadata = {"test": "test"}
    entity_without_metadata = entity.without_metadata()
    assert entity_without_metadata.metadata == {}


def test_entity_to_from_dict(emails_exp_1_dir_path: Path):
    """Test converting an entity to and from a dictionary."""
    entities = list(EntityUtilities.load_entities(emails_exp_1_dir_path / "extracted_entities"))
    entity = entities[0]

    # test that _eq__ method works (that attributes are created)
    assert entity == entity  # noqa: PLR0124

    entity_dict = entity.to_dict()
    new_entity = Entity.from_dict(entity_dict)

    assert entity == new_entity
    assert entity.properties == new_entity.properties

    evidence = entity.get_evidence_for_property_value("name", 0)
    assert evidence == ["email_1"]

    # test metadata
    entity.metadata["entity_id"] = "test_entity_id"
    entity.metadata["types"] = ["test_entity_type_1", "test_entity_type_2"]
    entity.metadata["split"] = "train"

    new_entity = Entity.from_json(entity.to_json())
    assert entity == new_entity

    new_entity_without_metadata = new_entity.without_metadata()
    assert new_entity_without_metadata.metadata == {}
