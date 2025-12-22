# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from dataclasses import dataclass
from typing import Any, ClassVar, override

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from kebab.contracts.entity import Entity, Property, PropertySchema, ValueType
from kebab.tasks.metrics.extraction.aesop import distances
from kebab.tasks.metrics.extraction.aesop.element_distance import ElementDistance


class ReferenceDistance(ElementDistance):
    """Distance class for reference properties.
    Computes distance based on matched entities.
    """

    def __init__(self, matched_entities: dict[str, dict[str, float]]):
        """Initialize reference distance."""
        self.matched_entities = matched_entities

    @override
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for reference distance."""
        if property_.data_type.value_type is not ValueType.REFERENCE:
            raise ValueError("Only ValueType.REFERENCE properties are supported.")

    @override
    def compute(self, value1: Any, value2: Any, property_: Property) -> float:
        """Compute distance between property values of ground truth and prediction entity."""
        ref1 = str(value1)
        ref2 = str(value2)
        if ref1 in self.matched_entities and ref2 in self.matched_entities[ref1]:
            return self.matched_entities[ref1][ref2]
        return 1.0

    @classmethod
    @override
    def build(
        cls,
        config: dict[str, Any],
        *,
        gt_entities: Any,
        pred_entities: Any,
        matching_info: Any,
        **kwargs: dict[str, Any],
    ) -> "ReferenceDistance":
        """Build reference distance from matched entities.

        Args:
            config: Configuration dictionary (not used).
            gt_entities: List of ground truth entities.
            pred_entities: List of predicted entities.
            matching_info: Matched pairs of entities.
            kwargs: Additional arguments
        Returns:
            ReferenceDistance instance.
        """
        matched_entities = {}
        for gt_idx, pred_idx in zip(matching_info.left_ind, matching_info.right_ind, strict=False):
            gt_entity = gt_entities[gt_idx]
            pred_entity = pred_entities[pred_idx]
            distance = matching_info.distances[gt_idx, pred_idx]
            gt_ref = str(gt_entity.entity_id)
            pred_ref = str(pred_entity.entity_id)
            if gt_ref not in matched_entities:
                matched_entities[gt_ref] = {}
            matched_entities[gt_ref][pred_ref] = distance
        return ReferenceDistance(matched_entities)


class ReferenceResolvingDistance(ElementDistance):
    """Reference resolving distance class.
    Resolves references in property values and computes distance on resolved values.
    """

    def __init__(
        self, internal_distance: ElementDistance, gt_entities: dict[str, Entity], pred_entities: dict[str, Entity]
    ):
        """Initialize reference resolving distance.

        Args:
            internal_distance: Element distance function to compute distance after resolving references.
            gt_entities: mapping of ground truth entity IDs to entities.
            pred_entities: mapping of predicted entity IDs to entities.
        """
        self.gt_entities = gt_entities
        self.pred_entities = pred_entities
        self.internal_distance = internal_distance

    @override
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for reference resolving distance."""
        if property_.data_type.value_type is not ValueType.REFERENCE:
            raise ValueError("Only ValueType.REFERENCE properties are supported.")

    def resolve_value(self, value: Any, is_gt: bool) -> Any:  # noqa: ANN401
        """Resolve reference to actual value."""
        entities = self.gt_entities if is_gt else self.pred_entities
        if entities is None:
            raise ValueError("Entities must be provided for reference resolving.")
        entity = entities.get(str(value))
        if entity is None:
            raise ValueError(f"Reference resolution failed: entity with id '{value}' not found.")
        return entity.properties["name"]

    @override
    def compute(self, value1: Any, value2: Any, property_: Property) -> float:
        """Compute distance between property values after resolving references.
        Assumes that value1 and value2 are entity IDs.
        And value1 is from ground truth entity, value2 is from predicted entity.
        """
        resolved_value1 = self.resolve_value(value1, is_gt=True)
        resolved_value2 = self.resolve_value(value2, is_gt=False)
        min_distance = 1.0
        for internal_value1 in enumerate(resolved_value1):
            for internal_value2 in enumerate(resolved_value2):
                distance = self.internal_distance(
                    internal_value1, internal_value2, property_, override_constraints=True
                )
                min_distance = min(min_distance, distance)
        return min_distance

    @classmethod
    @override
    def build(
        cls, config: dict[str, Any], *, gt_entities: Any, pred_entities: Any, **kwargs: dict[str, Any]
    ) -> "ReferenceResolvingDistance":
        """Create reference resolving distance from dictionary.

        Args:
            config: Configuration dictionary containing "internal_distance".
                "internal_distance": Configuration for the internal element distance function.
            gt_entities: List of ground truth entities.
            pred_entities: List of predicted entities.
            kwargs: Additional arguments for internal distance function.

        Returns:
            ReferenceResolvingDistance instance.
        """
        internal_distance_config = config.get("internal_distance", {})
        gt_entities_map = {entity.entity_id: entity for entity in gt_entities}
        pred_entities_map = {entity.entity_id: entity for entity in pred_entities}
        return ReferenceResolvingDistance(
            get_element_distance(
                internal_distance_config, **{"gt_entities": gt_entities, "pred_entities": pred_entities, **kwargs}
            ),
            gt_entities_map,
            pred_entities_map,
        )


PropertyDistanceValue = tuple[list[float], int]
PropertyScoreValue = tuple[list[float], int]


def get_element_distance(element_distance_params: dict[str, Any], **kwargs: dict[str, Any]) -> ElementDistance:
    """Get element distance function from parameters."""
    element_distance_cls_name = element_distance_params["name"]
    cls = getattr(distances, element_distance_cls_name, globals().get(element_distance_cls_name))
    if cls is None:
        raise ValueError(f"Element distance class '{element_distance_cls_name}' not found.")
    return cls.build(element_distance_params.get("params", {}), **kwargs)


@dataclass
class ValueMatchingRecordWithDistances:
    """Value matching record containing distances between matched values.
    Distances are between 0 and 1.
    The lower the distance, the more similar the values are.
    """

    gt_values: list[Any]
    pred_values: list[Any]
    matched_distances: list[float]
    unmatched_gt_values: list[Any]
    unmatched_pred_values: list[Any]


@dataclass
class ValueMatchingRecordWithScores:
    """Value matching record with scores for pairs of matched values.
    Scores are between 0 and 1.
    The higher the score, the more similar the values are.
    """

    gt_values: list[Any]
    pred_values: list[Any]
    matched_scores: list[float]
    unmatched_gt_values: list[Any]
    unmatched_pred_values: list[Any]


class PropertyDistance(abc.ABC):
    """Property distance class."""

    @abc.abstractmethod
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for property distance."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> ValueMatchingRecordWithDistances:
        """Compute distance between two entities for a given property id.

        Returns:
            - list of distances between matched property values of ground truth and prediction entity
            - number of unmatched property values in prediction entity"
        """
        raise NotImplementedError

    def __call__(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> ValueMatchingRecordWithDistances:
        """Compute distance between two entities for a given property id.

        Returns:
            - list of distances between matched property values of ground truth and prediction entity
            - number of unmatched property values in prediction entity"
        """
        self.check_constraints(property_)
        return self.compute(gt_entity, pred_entity, property_)

    @classmethod
    @abc.abstractmethod
    def build(cls, config: dict[str, Any], **kwargs: dict[str, Any]) -> "PropertyDistance":
        """Update internal distance functions with entity information for reference resolving distances."""
        raise NotImplementedError


class SingleValuePropertyDistance(PropertyDistance):
    """Single value property distance class."""

    def __init__(self, element_distance: ElementDistance):
        """Initialize single value property distance."""
        self.element_distance = element_distance

    @override
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for single value property distance."""
        if property_.is_collection:
            raise ValueError("Collection properties are not supported.")

    @override
    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> ValueMatchingRecordWithDistances:
        """Compute distance between two entities for a given property id.

        Args:
            gt_entity: Ground truth entity.
            pred_entity: Predicted entity.
            property_: Property to compute distance for.

        Returns:
            ValueMatchingRecordWithDistances containing list of distances between matched property values
        """
        gt_values = gt_entity[property_] if property_ in gt_entity else []  # noqa: SIM401
        pred_values = pred_entity[property_] if property_ in pred_entity else []  # noqa: SIM401
        if property_ not in gt_entity or property_ not in pred_entity:
            score = 1
        else:
            score = self.element_distance(gt_entity[property_][0], pred_entity[property_][0], property_)
        return ValueMatchingRecordWithDistances(gt_values, pred_values, [score], [], [])

    @classmethod
    @override
    def build(cls, config: dict[str, Any], **kwargs: dict[str, Any]) -> "SingleValuePropertyDistance":
        """Create single value property distance from config and additional arguments."""
        return SingleValuePropertyDistance(get_element_distance(config, **kwargs))


class SetPropertyDistance(PropertyDistance):
    """Set distance class."""

    def __init__(self, element_distance: ElementDistance):
        """Initialize set distance."""
        self.element_distance = element_distance

    @override
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for set distance."""
        return None

    def compute(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> ValueMatchingRecordWithDistances:
        """Compute distances between two sets of values for a given property.

        Args:
            gt_entity: Ground truth entity.
            pred_entity: Predicted entity.
            property_: Property to compute distances for.

        Returns:
            ValueMatchingRecordWithDistances containing distances between matched property values
        """
        gt_values = np.array(gt_entity[property_] if property_ in gt_entity else [])  # noqa: SIM401
        pred_values = np.array(pred_entity[property_] if property_ in pred_entity else [])  # noqa: SIM401
        if property_ not in gt_entity or property_ not in pred_entity:
            return ValueMatchingRecordWithDistances([], [], [1], list(gt_values), list(pred_values))

        distances = np.zeros((len(gt_values), len(pred_values)))
        for i, value1 in enumerate(gt_values):
            for j, value2 in enumerate(pred_values):
                distances[i, j] = self.element_distance(value1, value2, property_)
        result = match_items(distances)
        matched_scores = list(distances[result.left_ind, result.right_ind])
        unmatched_gt_indices = list(set(range(len(gt_values))) - set(result.left_ind))
        unmatched_pred_indices = list(set(range(len(pred_values))) - set(result.right_ind))
        return ValueMatchingRecordWithDistances(
            list(gt_values[result.left_ind]),
            list(pred_values[result.right_ind]),
            list(matched_scores),
            list(gt_values[unmatched_gt_indices]),
            list(pred_values[unmatched_pred_indices]),
        )

    @classmethod
    @override
    def build(cls, config: dict[str, Any], **kwargs: dict[str, Any]) -> "SetPropertyDistance":
        """Create set property distance from config and additional arguments."""
        return SetPropertyDistance(get_element_distance(config, **kwargs))


class PropertyScore:
    """Property score class."""

    def __init__(self, property_distance: PropertyDistance):
        """Initialize property score."""
        self.property_distance = property_distance

    def __call__(self, gt_entity: Entity, pred_entity: Entity, property_: Property) -> ValueMatchingRecordWithScores:
        """Compute the score for a given property.

        Args:
            gt_entity: Ground truth entity.
            pred_entity: Predicted entity.
            property_: Property to compute score for.

        Returns:
            ValueMatchingRecordWithScores containing scores for matched property values
        """
        value_matching_record = self.property_distance(gt_entity, pred_entity, property_)
        for distance in value_matching_record.matched_distances:
            if distance < 0 or distance > 1:
                raise ValueError(f"Distance value {distance} is out of range [0, 1].")
        return ValueMatchingRecordWithScores(
            value_matching_record.gt_values,
            value_matching_record.pred_values,
            [1.0 - d for d in value_matching_record.matched_distances],
            value_matching_record.unmatched_gt_values,
            value_matching_record.unmatched_pred_values,
        )

    @classmethod
    def build(cls, config: dict[str, Any], **kwargs: dict[str, Any]) -> "PropertyScore":
        """Create property score from configuration dictionary and additional arguments."""
        property_distance_cls = config.get("property_distance_cls")
        if property_distance_cls is None:
            raise ValueError("property_distance_cls must be provided in the config.")
        element_distance_config = config.get("element_distance", {})
        return PropertyScore(property_distance_cls.build(element_distance_config, **kwargs))


class EntityDistance:
    """Distance between two entities.

    Computed as a weighted sum of distances between properties.
    """

    DEFAULT_ELEMENT_DISTANCE_CONFIG: ClassVar[dict[str, Any]] = {"name": "TokenDistance"}
    DEFAULT_PROPERTY_WEIGHT = 1.0

    def __init__(
        self,
        property_schema: PropertySchema,
        property_to_distance_function_and_weight: dict[str, tuple[PropertyDistance, float]],
        default_simple_property_distance: PropertyDistance,
        default_reference_property_distance: PropertyDistance,
        default_property_weight: float = DEFAULT_PROPERTY_WEIGHT,
    ):
        """Initialize entity distance.

        Args:
            property_schema: Schema defining properties and their data types.
            property_to_distance_function_and_weight: Mapping of property names to distance functions and weights.
            default_simple_property_distance: Default distance function for non-reference properties not explicitly configured.
            default_reference_property_distance: Default distance function for reference properties not explicitly configured.
            default_property_weight: Default weight for properties not explicitly configured.
        """
        self.property_to_distance_function_and_weight = property_to_distance_function_and_weight
        self.property_schema = property_schema
        self.default_simple_property_distance = default_simple_property_distance
        self.default_reference_property_distance = default_reference_property_distance
        self.default_property_weight = default_property_weight
        self.__normalize_weights()

    def __call__(self, gt_entity: Entity, pred_entity: Entity) -> float:
        """Compute distance between two entities."""
        distance = 0.0
        for property_id, (property_distance, weight) in self.property_to_distance_function_and_weight.items():
            value_matching_record = property_distance(
                gt_entity, pred_entity, self.property_schema.properties[property_id]
            )
            distance += float(np.mean(value_matching_record.matched_distances)) * weight
        return distance

    def __normalize_weights(self):
        """Normalize weights to sum to 1."""
        weight_sum = 0.0
        for property_id in self.property_schema.properties:
            if property_id not in self.property_to_distance_function_and_weight:
                weight_sum += self.default_property_weight
            else:
                weight_sum += self.property_to_distance_function_and_weight[property_id][1]
        for property_id in self.property_schema.properties:
            if property_id not in self.property_to_distance_function_and_weight:
                self.property_to_distance_function_and_weight[property_id] = (
                    self.default_simple_property_distance
                    if self.property_schema.properties[property_id].data_type.value_type != ValueType.REFERENCE
                    else self.default_reference_property_distance,
                    self.default_property_weight / weight_sum,
                )
            else:
                self.property_to_distance_function_and_weight[property_id] = (
                    self.property_to_distance_function_and_weight[property_id][0],
                    self.property_to_distance_function_and_weight[property_id][1] / weight_sum,
                )

    @classmethod
    def build(
        cls, config: dict[str, Any], property_schema: PropertySchema, **kwargs: dict[str, Any]
    ) -> "EntityDistance":
        """Create entity distance from dictionary  and additional arguments."""
        property_to_distance_function_and_weight = {}
        default_weight = config.get("default_property_weight", cls.DEFAULT_PROPERTY_WEIGHT)
        for property_id, params in config["property_to_distance"].items():
            if property_id not in property_schema.properties:
                raise ValueError(f"Property '{property_id}' is not in the property schema.")
            element_distance_config = (
                {"name": "ReferenceResolvingDistance", "params": {"internal_distance": params["distance_function"]}}
                if property_schema.properties[property_id].data_type.value_type == ValueType.REFERENCE
                else params["distance_function"]
            )
            property_distance = (
                SetPropertyDistance.build(element_distance_config, **kwargs)
                if property_schema.properties[property_id].is_collection
                else SingleValuePropertyDistance.build(element_distance_config, **kwargs)
            )
            weight = params.get("weight", default_weight)
            property_to_distance_function_and_weight[property_id] = (property_distance, weight)
        default_element_distance_config = config.get("default_property_distance", cls.DEFAULT_ELEMENT_DISTANCE_CONFIG)
        default_reference_element_distance_config = {
            "name": "ReferenceResolvingDistance",
            "params": {"internal_distance": default_element_distance_config},
        }
        default_simple_property_distance = SetPropertyDistance.build(default_element_distance_config, **kwargs)
        default_reference_property_distance = SetPropertyDistance.build(
            default_reference_element_distance_config, **kwargs
        )
        return EntityDistance(
            property_schema,
            property_to_distance_function_and_weight,
            default_simple_property_distance,
            default_reference_property_distance,
            default_weight,
        )


@dataclass
class MatchingInfo:
    """
    Stores information about matched pairs of entities.

    Attributes:
        left_ind: the indices of the matched entities in the ground truth
        right_ind: the indices of the matched entities in the predictions
        left_unmatched: the indices of the unmatched entities in the ground truth
        right_unmatched: the indices of the unmatched entities in the predictions
        distances: the distances between the matched entities

    """

    left_ind: list[int]
    right_ind: list[int]
    left_unmatched: list[int]
    right_unmatched: list[int]
    distances: np.ndarray


def match_items(distances: np.ndarray, threshold: float = 1.0) -> MatchingInfo:
    """Match items from two sets of items based on a distance matrix if the distance is within a threshold.

    Args:
        distances: 2D array where distances[i,j] is the distance between item i and item j.
        threshold: Maximum distance threshold for considering items as matched.

    Returns:
        MatchingInfo object containing indices of matched and unmatched items.
    """
    epsilon = 0.01
    left_ind, right_ind = min_weight_full_bipartite_matching(csr_matrix(distances + epsilon))

    matched_dists = distances[left_ind, right_ind]
    indices_within_threshold = np.where(matched_dists <= threshold)
    unmatched_ind = np.where(matched_dists > threshold)
    left_ind_matched = left_ind[indices_within_threshold]
    right_ind_matched = right_ind[indices_within_threshold]
    left_ind_unmatched = left_ind[unmatched_ind]
    right_ind_unmatched = right_ind[unmatched_ind]
    return MatchingInfo(left_ind_matched, right_ind_matched, left_ind_unmatched, right_ind_unmatched, distances)
