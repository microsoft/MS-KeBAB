import abc
from typing import Any

from kebab.contracts.entity import Property


class ElementDistance(abc.ABC):
    """Element distance class.
    Element distance values should be between 0 and 1.
    """

    @abc.abstractmethod
    def check_constraints(self, property_: Property) -> None:
        """Check constraints for element distance."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self, value1: Any, value2: Any, property_: Property) -> float:  # noqa: ANN401
        """Compute distance between property values of ground truth and prediction entity."""
        raise NotImplementedError

    def __call__(self, value1: Any, value2: Any, property_: Property, override_constraints: bool = False) -> float:  # noqa: ANN401
        """Compute distance between property values of ground truth and prediction entity."""
        if not override_constraints:
            self.check_constraints(property_)
        return self.compute(value1, value2, property_)

    @classmethod
    def build(cls, config: dict[str, Any], **kwargs: dict[str, Any]) -> "ElementDistance":
        """Build element distance from configuration dictionary and additional arguments.

        Args:
            config: Configuration dictionary.
            kwargs: Additional arguments.

        Returns:
            ElementDistance instance.
        """
        raise NotImplementedError
