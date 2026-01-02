from typing import Any

from kebab.contracts.entity import PropertySchema, ValueType
from kebab.tasks.metrics.extraction.aesop.property_score import (
    EntityDistance,
    PropertyScore,
    SetPropertyDistance,
    SingleValuePropertyDistance,
)


class MetricsFactory:
    """Factory for creating distance and scoring functions for AESOP metrics."""

    def __init__(self, config: dict[str, Any], property_schema: PropertySchema) -> None:
        """Create MetricsFactory from metrics config.

        Args:
            config: deserialized metrics config.
            property_schema: Property schema for the entities.
        """
        self.properties_to_skip = set(config.get("properties_to_skip", []))
        self.property_schema = property_schema
        self.property_distance_configs = config["property_distance_functions"]
        self.default_property_distance_config = config["default_property_distance"]
        self.entity_distance_config = config["entity_distance"]

    def get_score_for_property(
        self,
        property_id: str,
        *,
        gt_entities: Any,  # noqa: ANN401
        pred_entities: Any,  # noqa: ANN401
        matching_info: Any,  # noqa: ANN401
    ) -> PropertyScore | None:
        """Get property score function for given property.

        Args:
            property_id: Property identifier.
            gt_entities: Ground truth entities.
            pred_entities: Predicted entities.
            matching_info: Matched pairs of entities.
        """
        if property_id in self.properties_to_skip:
            return None
        params = self.property_distance_configs.get(property_id, self.default_property_distance_config)
        element_distance_config = (
            {"name": "ReferenceDistance", "params": params}
            if property_id not in self.property_schema.properties
            or self.property_schema.properties[property_id].data_type.value_type == ValueType.REFERENCE
            else params
        )
        property_distance_cls = (
            SetPropertyDistance
            if property_id not in self.property_schema.properties
            or self.property_schema.properties[property_id].is_collection
            else SingleValuePropertyDistance
        )
        return PropertyScore.build(
            {"property_distance_cls": property_distance_cls, "element_distance": element_distance_config},
            gt_entities=gt_entities,
            pred_entities=pred_entities,
            matching_info=matching_info,
        )

    def get_matching_score_function(self, *, gt_entities: Any, pred_entities: Any) -> EntityDistance:  # noqa: ANN401
        """Get entity matching score function in the given context.

        Args:
            gt_entities: Ground truth entities.
            pred_entities: Predicted entities.
        """
        return EntityDistance.build(
            config=self.entity_distance_config,
            property_schema=self.property_schema,
            gt_entities=gt_entities,
            pred_entities=pred_entities,
        )
