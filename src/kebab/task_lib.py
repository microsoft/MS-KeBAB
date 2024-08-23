# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Task and instance interface implementation for MS-KeBAB benchmark."""

from __future__ import annotations
from pathlib import Path
from abc import abstractmethod

class Task:
  """Represents a benchmark task with its task instances."""

  _name: str

  @property
  def name(self) -> str:
    return self._name

  @property
  @abstractmethod
  def instances(self):
    raise NotImplementedError()

class ExtractionTask(Task):
  """Represents an extraction benchmark task with its task instances."""

  _instances: dict[str, ExtractionTaskInstance]

  @property
  def instances(self) -> dict[str, ExtractionTaskInstance]:
    return self._instances

  def __init__(self):
    self._name = "Extraction"
    self._instances = {}

  def add_instance(self, instance: ExtractionTaskInstance):
    """Add an extraction task instance."""
    assert instance.name not in self._instances
    self._instances[instance.name] = instance

  def add_instances(self, instances: dict[str, ExtractionTaskInstance]):
    """Add extraction task instances."""
    for name, instance in instances.items():
      assert name not in self._instances
      self._instances[name] = instance

  def evaluate(self, extracted_entities: Path, ground_truth_extracted_entities: Path) -> dict[str, float]:
    """Evaluate an output for the extraction task."""
  
    # TODO: Implement actual metric computation
    return {
      "primary_extraction_metric": 0.8,
      "secondary_extraction_metric": 0.6
    }

class LinkingTask(Task):
  """Represents a linking benchmark task with its task instances."""

  _instances: dict[str, LinkingTaskInstance]

  @property
  def instances(self) -> dict[str, LinkingTaskInstance]:
    return self._instances

  def __init__(self):
    self._name = "Linking"
    self._instances = {}

  def add_instance(self, instance: LinkingTaskInstance):
    """Add a linking task instance."""
    assert instance.name not in self._instances
    self._instances[instance.name] = instance

  def add_instances(self, instances: dict[str, LinkingTaskInstance]):
    """Add linking task instances."""
    for name, instance in instances.items():
      assert name not in self._instances
      self._instances[name] = instance

  def evaluate(self, entity_fragment_pairs: Path, ground_truth_boolean: Path) -> dict[str, float]:
    """Evaluate an output for the linking task."""
  
    # TODO: Implement actual metric computation
    return {
      "primary_linking_metric": 0.8,
      "secondary_linking_metric": 0.6
    }

class TaskInstance:
  """Represents a benchmark task instance with its data files."""

  _name: str
  _heldout: bool

  @property
  def name(self) -> str:
    return self._name

  @property
  def heldout(self) -> bool:
    return self._heldout

  @property
  def parent(self) -> Task:
    raise NotImplementedError()

class ExtractionTaskInstance(TaskInstance):
  """Represents an extraction benchmark task instance with its data files."""

  _parent: ExtractionTask
  _data_train_extracts: Path
  _data_train_ground_truth_extracted_entities: Path
  _data_dev_extracts: Path
  _data_dev_ground_truth_extracted_entities: Path
  _data_test_extracts: Path
  _data_test_ground_truth_extracted_entities: Path

  @property
  def parent(self) -> ExtractionTask:
    return self._parent

  @property
  def data_train_extracts(self) -> Path:
    return self._data_train_extracts

  @property
  def data_train_ground_truth_extracted_entities(self) -> Path:
    return self._data_train_ground_truth_extracted_entities

  @property
  def data_dev_extracts(self) -> Path:
    return self._data_dev_extracts

  @property
  def data_dev_ground_truth_extracted_entities(self) -> Path:
    return self._data_dev_ground_truth_extracted_entities

  @property
  def data_test_extracts(self) -> Path:
    return self._data_test_extracts

  @property
  def data_test_ground_truth_extracted_entities(self) -> Path:
    if self._heldout:
      raise NotImplementedError()
    return self._data_test_ground_truth_extracted_entities

  def __init__(self, name: str, data_dict: dict[str, str], parent: ExtractionTask, heldout: bool):
    """Initialize an extraction task instance."""
    self._name = name
    self._parent = parent
    self._heldout = heldout
    assert len(data_dict) >= 5 and len(data_dict) <= 6 # number of expected data files
    self._data_train_extracts = Path(data_dict["train_extracts"])
    self._data_train_ground_truth_extracted_entities = Path(data_dict["train_ground_truth_extracted_entities"])
    self._data_dev_extracts = Path(data_dict["dev_extracts"])
    self._data_dev_ground_truth_extracted_entities = Path(data_dict["dev_ground_truth_extracted_entities"])
    self._data_test_extracts = Path(data_dict["test_extracts"])
    if not self._heldout:
      self._data_test_ground_truth_extracted_entities = Path(data_dict["test_ground_truth_extracted_entities"])

  def evaluate(self, extracted_entities: Path, set_type: str) -> dict[str, float]:
    """Evaluate an output for the extraction task instance."""
    if set_type == "train":
      ground_truth_extracted_entities = self.data_train_ground_truth_extracted_entities
    elif set_type == "dev":
      ground_truth_extracted_entities = self.data_dev_ground_truth_extracted_entities
    elif set_type == "test":
      if self._heldout:
        raise ValueError("Cannot evaluate because the ground truth for the test set for this extraction task is withheld")
      else:
        ground_truth_extracted_entities = self.data_test_ground_truth_extracted_entities
    else:
      raise ValueError("Unknown set type")
    return self._parent.evaluate(extracted_entities=extracted_entities,
                                 ground_truth_extracted_entities=ground_truth_extracted_entities)

class LinkingTaskInstance(TaskInstance):
  """Represents an linking benchmark task instance with its data files."""

  _parent: LinkingTask
  _data_train_entity_fragment_pairs: Path
  _data_train_ground_truth_boolean: Path
  _data_dev_entity_fragment_pairs: Path
  _data_dev_ground_truth_boolean: Path
  _data_test_entity_fragment_pairs: Path
  _data_test_ground_truth_boolean: Path

  @property
  def name(self) -> str:
    return self._name

  @property
  def parent(self) -> LinkingTask:
    return self._parent

  @property
  def data_train_entity_fragment_pairs(self) -> Path:
    return self._data_train_entity_fragment_pairs

  @property
  def data_train_ground_truth_boolean(self) -> Path:
    return self._data_train_ground_truth_boolean

  @property
  def data_dev_entity_fragment_pairs(self) -> Path:
    return self._data_dev_entity_fragment_pairs

  @property
  def data_dev_ground_truth_boolean(self) -> Path:
    return self._data_dev_ground_truth_boolean

  @property
  def data_test_entity_fragment_pairs(self) -> Path:
    return self._data_test_entity_fragment_pairs

  @property
  def data_test_ground_truth_boolean(self) -> Path:
    if self._heldout:
      raise NotImplementedError()
    return self._data_test_ground_truth_boolean

  def __init__(self, name: str, data_dict: dict[str, str], parent: LinkingTask, heldout: bool):
    """Initialize an linking task instance."""
    self._name = name
    self._parent = parent
    self._heldout = heldout
    assert len(data_dict) >= 5 and len(data_dict) <= 6 # number of expected data files
    self._data_train_entity_fragment_pairs = Path(data_dict["train_entity_fragment_pairs"])
    self._data_train_ground_truth_boolean = Path(data_dict["train_ground_truth_boolean"])
    self._data_dev_entity_fragment_pairs = Path(data_dict["dev_entity_fragment_pairs"])
    self._data_dev_ground_truth_boolean = Path(data_dict["dev_ground_truth_boolean"])
    self._data_test_entity_fragment_pairs = Path(data_dict["test_entity_fragment_pairs"])
    if not self._heldout:
      self._data_test_ground_truth_boolean = Path(data_dict["test_ground_truth_boolean"])

  def evaluate(self, entity_fragment_pairs: Path, set_type: str) -> dict[str, float]:
    """Evaluate an output for the linking task instance."""
    if set_type == "train":
      ground_truth_boolean = self.data_train_ground_truth_boolean
    elif set_type == "dev":
      ground_truth_boolean = self.data_dev_ground_truth_boolean
    elif set_type == "test":
      if self._heldout:
        raise ValueError("Cannot evaluate because the ground truth for the test set for this linking task is withheld")
      else:
        ground_truth_boolean = self.data_test_ground_truth_boolean
    else:
      raise ValueError("Unknown set type")
    return self._parent.evaluate(entity_fragment_pairs=entity_fragment_pairs,
                                 ground_truth_boolean=ground_truth_boolean)
