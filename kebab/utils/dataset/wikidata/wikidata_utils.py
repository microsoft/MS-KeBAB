# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pyright: reportReturnType=false, reportOptionalMemberAccess=false
# ruff: noqa: TRY400, S113, PLR2004, RET505, B008, A001, PLW2901, PLR1714, ARG005
"""Utilities for extracting from Wikidata."""

from __future__ import annotations

import json
import logging
import pathlib
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum

import requests
from tqdm import tqdm


REFERENCE_REGEX_PATTERN: re.Pattern = re.compile(r"Q\d+")


class TypeProperties(Enum):
    """Enumeration of the Wikidata type related properties."""

    INSTANCE_OF = "P31"
    SAME_AS = "P460"
    SUBCLASS_OF = "P279"
    PART_OF = "P361"
    FACET_OF = "P1269"
    SUB_PROPERTY_OF = "P1647"


@dataclass
class SimpleEntity:
    """A simple representation of a Wikidata entity."""

    id: str
    name: str
    description: str | None = None
    aliases: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    properties: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self, minimal: bool = True) -> dict:
        """Convert the SimpleEntity to a dictionary."""
        d = asdict(self)
        if minimal:
            for p in ["description", "aliases", "types", "properties"]:
                if not d[p]:
                    del d[p]

        return d

    @classmethod
    def from_dict(cls, data: dict) -> SimpleEntity:
        """Create a SimpleEntity from a dictionary."""
        return SimpleEntity(**data)

    def to_json(self, minimal: bool = True) -> str:
        """Convert the SimpleEntity to a JSON string."""
        return json.dumps(self.to_dict(minimal=minimal), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> SimpleEntity:
        """Create a SimpleEntity from a JSON string."""
        return SimpleEntity.from_dict(json.loads(data))

    @classmethod
    def from_wikidata_entity(
        cls,
        entity: dict,
        allowed_properties: Iterable[str] | None = None,
        prohibited_qualifiers: Iterable[str] | None = None,
        include_descriptions: bool = True,
    ) -> SimpleEntity:
        """Convert a Wikidata entity to a simpler representation."""
        allowed_properties = set(allowed_properties) if allowed_properties is not None else None

        if allowed_properties is not None:
            allowed_properties.add(TypeProperties.INSTANCE_OF.value)

        entity_id = entity["id"]
        label = entity.get("labels", {}).get("en", {}).get("value", "")

        description = entity.get("descriptions", {}).get("en", {}).get("value", "")
        aliases = [v["value"] for v in entity.get("aliases", {}).get("en", {})]

        # extract properties
        claims = entity.get("claims", {})
        keys = set(claims.keys())
        if allowed_properties is not None:
            keys &= allowed_properties

        ent_properties = {}

        for prop in keys:
            prop_claims = claims.get(prop, [])
            prop_values = []

            for claim in prop_claims:
                if prohibited_qualifiers:
                    qualifiers = claim.get("qualifiers", {})
                    if any(q in qualifiers for q in prohibited_qualifiers):
                        continue

                mainsnak = claim.get("mainsnak", {})
                if mainsnak.get("rank") == "deprecated":
                    continue

                if mainsnak.get("snaktype") == "value":
                    try:
                        prop_values.append(mainsnak.get("datavalue", {}).get("value", {}).get("id"))
                    except AttributeError:
                        prop_values.append(mainsnak.get("datavalue", {}).get("value"))

            ent_properties[prop] = prop_values

        # store entity types explicitly
        types = ent_properties.get(TypeProperties.INSTANCE_OF.value, [])
        ent_properties.pop(TypeProperties.INSTANCE_OF.value, None)

        simple_entity = SimpleEntity(
            id=entity_id,
            name=label,
            description=description if include_descriptions else None,
            aliases=list(set(aliases)),
            types=list(set(types)),
            properties=ent_properties,
        )

        return simple_entity


def _query_entities_via_api(wikidata_ids: Iterable[str], english_only: bool = True) -> dict | None:
    """
    Query Wikidata API for a given list of IDs and return the dictionary of the corresponding objects.

    Args:
        wikidata_ids: Iterable[str], the IDs to query, e.g., ['P19'] for place of birth.
        english_only: bool, whether to return only English labels.

    Returns:
        A dictionary containing the Wikidata information for the given IDs.
    """
    wikidata_ids = list(wikidata_ids)

    # make the request to the Wikidata API
    url = "https://www.wikidata.org/w/api.php"
    params = {"action": "wbgetentities", "ids": "|".join(wikidata_ids), "format": "json"}

    if english_only:
        params["languages"] = "en"

    try:
        response = requests.get(url, params=params)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Wikidata for IDs {wikidata_ids}: {e}")
        return None

    if response.status_code == 200:
        result = response.json()
        if "entities" not in result:
            logging.error(f"Error querying Wikidata for IDs {wikidata_ids}: {result}")
            return None

        entities = response.json()["entities"]

        if len(entities) != len(wikidata_ids):
            logging.warning(f"Error querying Wikidata for IDs {wikidata_ids}: {entities}")

        for entity_id, entity in entities.items():
            if entity_id != entity["id"]:
                logging.warning(f"Received redirect for Wikidata ID {entity_id} -> {entity['id']}")

        return entities
    else:
        logging.error(f"Error querying Wikidata for IDs {wikidata_ids}: {response.text}")
        return None


def query_entities_via_api(
    wikidata_ids: Iterable[str],
    english_only: bool = True,
    batch_size: int = 50,
) -> dict | None:
    """
    Query Wikidata API for a given list of IDs and return the dictionary of the corresponding objects.

    Args:
        wikidata_ids: Iterable[str], the IDs to query, e.g., ['P19'] for place of birth.
        english_only: bool, whether to return only English labels.
        batch_size: int, the number of IDs to query in each batch.

    Returns:
        A dictionary containing the Wikidata information for the given IDs.
    """
    wikidata_ids = list(wikidata_ids)
    logging.info(f"Querying {len(wikidata_ids):,d} items from Wikidata...")

    entities = {}
    for i in range(0, len(wikidata_ids), batch_size):
        batch_ids = wikidata_ids[i : i + batch_size]
        batch_entities = _query_entities_via_api(batch_ids, english_only=english_only)
        if batch_entities:
            entities.update(batch_entities)

        if ((i + batch_size) // batch_size) % 10 == 0:
            logging.info(f"Received {len(entities):,d} out of {len(wikidata_ids):,d} entities via API")

    return entities


def query_single_entity_via_api(wikidata_id: str, english_only: bool = True) -> dict | None:
    """Query Wikidata for a given ID and return the dictionary of the corresponding object."""
    entities = query_entities_via_api([wikidata_id], english_only=english_only)
    return entities.get(wikidata_id) if entities else None


def scrape_properties_via_api(
    start: int = 1,
    end: int = 15_000,
    stop_after_contiguous_errors: int = 5,
    save_every: int = 1000,
    output_path: pathlib.Path | None = None,
) -> None:
    """
    Query the IDs and English labels for a range of properties via the Wikidata API.

    The Wikidata API is rate-limited, so this method will take approx. 3 hours to complete for full range.
    The method will query the Wikidata API for properties with IDs from `start` to `end` (inclusive),
    and save the results to a JSON file. The method will stop after encountering `stop_after_contiguous_errors`
    errors in a row.
    (We could batch requests to speed up the process, but not doing so since this method is rarely used.)
    """
    logging.info(f"Querying {end - start + 1} properties from Wikidata...")

    output_path = output_path or pathlib.Path.cwd() / f"wikidata_properties_{start}_{end}.json"

    properties = {}
    error_count = 0
    cont_error_count = 0

    for i in tqdm(range(start, end + 1)):
        item = query_entities_via_api(f"P{i}")

        if not item:
            logging.warning(f"Error querying Wikidata for ID P{i}, retrying...")
            time.sleep(60)
            item = query_single_entity_via_api(f"P{i}")

        if item:
            cont_error_count = 0
            property = {
                "label": item.get("labels", {}).get("en", {}).get("value", ""),
                "description": item.get("descriptions", {}).get("en", {}).get("value", ""),
                "aliases": [v["value"] for v in item.get("aliases", {}).get("en", [])],
            }

            if not property["label"]:
                continue

            properties[f"P{i}"] = property
        else:
            logging.warning(f"Error querying Wikidata for ID P{i}")
            error_count += 1
            cont_error_count += 1

        if cont_error_count >= stop_after_contiguous_errors:
            logging.error(f"Stopping after {stop_after_contiguous_errors} contiguous errors")
            break

        if save_every > 0 and i % save_every == 0:
            with open(output_path, mode="w", encoding="utf-8") as f:
                json.dump(properties, f)

    if error_count:
        logging.warning(f"Encountered {error_count} errors while querying Wikidata")

    logging.info(f"Finished processing {end - start + 1} records, found {len(properties)} properties")
    for k, v in properties.items():
        logging.info(f"{k}: {v}")

    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(properties, f)


def extract_simple_entities_from_dump(
    wikidata_json_dump_path: pathlib.Path = pathlib.Path.cwd() / "latest-all.json",
    properties: Iterable[str] | None = None,
    input_entity_predicate: Callable[[dict], bool] | None = None,
    output_entity_predicate: Callable[[SimpleEntity], bool] | None = None,
    include_descriptions: bool = True,
) -> Iterable[SimpleEntity]:
    """Extract all items that match the predicate and their requested properties from the Wikidata JSON dump."""
    logging.info(f"Extracting entities from {wikidata_json_dump_path}...")

    processed_count = 0
    error_count = 0
    skipped_count = 0

    properties = set(properties) if properties else None
    input_entity_predicate = input_entity_predicate or (lambda x: True)
    output_entity_predicate = output_entity_predicate or (lambda x: True)

    with open(wikidata_json_dump_path, encoding="utf-8") as f:
        for line in f:
            processed_count += 1
            if processed_count % 500_000 == 0:
                logging.info(f"Processed {processed_count} records")

            # the json is a single array, not json-lines, but we are loading it line by line
            line = line.strip()
            if line == "[" or line == "]":
                continue

            if line.endswith(","):
                line = line[:-1]

            try:
                entity = json.loads(line.strip())

                # check if the input predicate applies
                if not input_entity_predicate(entity):
                    continue

                ent_type = entity.get("type")
                label = entity.get("labels", {}).get("en", {}).get("value", "")

                if not label or not ent_type:
                    skipped_count += 1
                    continue

                simple_entity = SimpleEntity.from_wikidata_entity(
                    entity,
                    allowed_properties=properties,
                    include_descriptions=include_descriptions,
                )

                # check if the input predicate applies
                if not output_entity_predicate(simple_entity):
                    continue

                yield simple_entity

            except json.JSONDecodeError:
                logging.warning(f"Error parsing line: {line}")
                error_count += 1

    if error_count:
        logging.warning(f"Encountered {error_count} JSON parsing errors")

    if skipped_count:
        logging.warning(f"Skipped {skipped_count} entities without labels or type")


def load_properties(path: pathlib.Path) -> dict[str, dict]:
    """Load (previously extracted) Wikidata properties from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_type_hierarchy(type_hierarchy_path: pathlib.Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load the hierarchy of Wikidata types."""
    logging.info("Loading the hierarchy of Wikidata types.")
    graph = {}
    type_id_to_node = {}

    with open(type_hierarchy_path, encoding="utf-8") as f:
        for line in f:
            node = json.loads(line.strip())
            graph[node["id"]] = node
            for node_id in node["merged_ids"]:
                type_id_to_node[node_id] = node

            for node_id in node["redirect_from_ids"]:
                type_id_to_node[node_id] = node

    logging.info(f"Type hierarchy contains {len(graph):,d} nodes")

    return graph, type_id_to_node


def load_simple_entities(simple_entities_path: pathlib.Path) -> Iterable[dict]:
    """Load the list of simple entities extracted from Wikidata."""
    count = 0
    with open(simple_entities_path, encoding="utf-8") as f:
        for line in f:
            entity = json.loads(line.strip())
            count += 1
            yield entity

    logging.info(f"Read {count:,d} entities")


def collect_all_subtypes(graph: dict[str, dict], type_id: str) -> set[str]:
    """Collect all subtypes of the given type from the type hierarhcy."""
    type_id_to_node = {}
    for node in graph.values():
        for node_id in node["merged_ids"]:
            type_id_to_node[node_id] = node

        for node_id in node["redirect_from_ids"]:
            type_id_to_node[node_id] = node

    type_id = type_id_to_node[type_id]["id"]

    subtypes = set()
    stack = [type_id]
    while stack:
        current_type_id = stack.pop()

        if current_type_id not in graph:
            logging.warning(f"Type {current_type_id} not found in the graph")
            continue

        for merged_id in graph[current_type_id]["merged_ids"]:
            subtypes.add(merged_id)

        for redirect_id in graph[current_type_id]["redirect_from_ids"]:
            subtypes.add(redirect_id)

        logging.info(f"Added {current_type_id} ({graph[current_type_id]['name']}) to the subtypes")

        for child_id in graph[current_type_id]["children"]:
            stack.append(child_id)  # noqa: PERF402

    return subtypes


def collect_wikidata_simple_entities(
    ids_to_include: set[str],
    wikidata_simple_entities_path: pathlib.Path,
    query_api: bool = False,
    append_to_file: bool = False,
    include_properties: bool = False,
    include_descriptions: bool = False,
) -> dict[str, SimpleEntity]:
    """Get Wikidata simplified entities for the specified IDs from a file and query API for any missing entities ."""
    logging.info(f"Loading Wikidata simple entities from {wikidata_simple_entities_path}")

    required_ids = set(ids_to_include)
    entities = {}
    with open(wikidata_simple_entities_path, encoding="utf-8") as f:
        for line in f:
            entity = json.loads(line)

            if entity["id"] not in required_ids:
                continue

            entities[entity["id"]] = entity
            required_ids.remove(entity["id"])

    if required_ids:
        if query_api:
            logging.info(f"Querying {len(required_ids):,} entities via the Wikidata API")

            q_entities = query_entities_via_api(required_ids)

            if q_entities is None:
                logging.error("Failed to query entities via the Wikidata API.")
                return entities

            q_entities = {
                k: SimpleEntity.from_wikidata_entity(
                    e, allowed_properties=[] if include_properties else None, include_descriptions=include_descriptions
                )
                for k, e in q_entities.items()
                if e is not None
            }

            if append_to_file:
                with open(wikidata_simple_entities_path, mode="a", encoding="utf-8") as f:
                    for entity in q_entities.values():
                        f.write(json.dumps(entity, ensure_ascii=False) + "\n")
                logging.info(f"Appended {len(q_entities):,} entities to {wikidata_simple_entities_path}")

            entities.update(q_entities)
        else:
            logging.warning(f"Failed to find {len(required_ids):,} entities in the list of simple entities")

    logging.info(f"Found {len(entities):,} Wikidata entities.")

    return entities
