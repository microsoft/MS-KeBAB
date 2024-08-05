# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
import json
import logging
import pathlib
from typing import Any

from kebab.document import Document
from kebab.entity import Entity


def configure_logging(
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: str = "log",
    level: int = logging.INFO,
) -> None:
    """Configure logging for the application."""
    handlers: list[logging.Handler] = []

    if log_to_console:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        handlers.append(handler)

    if log_to_file:
        path = pathlib.Path.cwd() / log_dir / f"{datetime.datetime.now():log_%Y-%m-%d-%H-%M-%S}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)04d\t%(levelname)s\t%(name)s\t%(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        handlers=handlers,
        level=logging.NOTSET,
    )


class CustomEncoder(json.JSONEncoder):
    """JSON encoder for entities and documents."""

    def default(self, o: Any) -> Any:  # noqa: ANN401, D102
        if isinstance(o, Entity):
            return o.to_dict()
        if isinstance(o, Document):
            return o.to_dict()
        return super().default(o)
