from typing import Any
from dataclasses import dataclass, field

from kebab.tasks.metrics.extraction.calculator import MetricConfig
from kebab.tasks.metrics.extraction.aesop.distances import (
    EntityDistance,
    PropertyScore,
    SetPropertyDistance,
    SingleValuePropertyDistance,
)
from kebab.contracts.entity import PropertySchema, ValueType

@dataclass
class ValueAveragedAesopConfig(MetricConfig):
    """Value-averaged-AESOP metric configuration."""

    matching_threshold: float
    property_schema: PropertySchema
    property_distance_configs: dict[str, Any]
    default_property_distance_config: dict[str, Any]
    entity_distance_config: dict[str, Any]
    properties_to_skip: set[str] = field(default_factory=set)

    @staticmethod
    def build(config: dict[str, Any], property_schema: PropertySchema) -> "ValueAveragedAesopConfig":
        """Create value-averaged-AESOP metric configuration from dictionary."""

        properties_to_skip = set(config.get("properties_to_skip", []))

        return ValueAveragedAesopConfig(
            matching_threshold=config["matching_threshold"],
            property_schema=property_schema,
            properties_to_skip=properties_to_skip,
            property_distance_configs=config["property_distance_functions"],
            default_property_distance_config=config["default_property_distance"],
            entity_distance_config=config["entity_distance"]
        )
        
    
    def get_score_for_property(self, property_id: str, context: dict[str, Any]) -> PropertyScore | None:
        """Get property score function for given property in the context."""
        if property_id in self.properties_to_skip:
            return None
        params = self.property_distance_configs.get(property_id, self.default_property_distance_config)
        element_distance_config = ({"name": "ReferenceDistance", "params": params}
                if context["property_schema"].properties[property_id].data_type.value_type == ValueType.REFERENCE
                else params)
        property_distance_cls = (
                SetPropertyDistance
                if self.property_schema.properties[property_id].is_collection
                else SingleValuePropertyDistance
            )
        return PropertyScore.build({"property_distance_cls": property_distance_cls,
                                    "element_distance": element_distance_config}, context)

    def get_matching_score_function(self, context: dict[str, Any]) -> EntityDistance:
        """Get entity matching score function in the given context."""
        if "property_schema" not in context:
            context["property_schema"] = self.property_schema
        return EntityDistance.build(
            config=self.entity_distance_config,
            context=context,
        )