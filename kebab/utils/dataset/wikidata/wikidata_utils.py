# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pyright: reportReturnType=false, reportOptionalMemberAccess=false
# ruff: noqa: TRY400, S113, PLR2004, RET505, B008, PLW2901, PLR1714, ARG005
"""Utilities for extracting from Wikidata."""

from __future__ import annotations

import json
import logging
import pathlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import requests

from kebab.contracts.entity import Entity


ENTITY_REFERENCE_REGEX_PATTERN: re.Pattern = re.compile(r"Q\d+")


class TypeProperties(Enum):
    """Enumeration of the Wikidata type related properties."""

    INSTANCE_OF = "P31"
    SAME_AS = "P460"
    SUBCLASS_OF = "P279"
    PART_OF = "P361"
    FACET_OF = "P1269"
    SUB_PROPERTY_OF = "P1647"
    WIKIDATA_ITEM_OF_THIS_PROPERTY = "P1629"


@dataclass
class WikidataEntity(Entity):
    """
    A simplified representation of a Wikidata entity.
    - Name+Aliases are stored in the "name" property.
    - Description and Wikipedia title are stored in the "metadata" property.
    - Contains a property for easier access the types of the entity (which are stored in the properties under P31 key).
    """

    @property
    def name(self) -> str:
        """Return the name of the entity."""
        return self.properties["name"][0] if ("name" in self.properties and len(self.properties["name"]) > 0) else ""

    @property
    def aliases(self) -> list[str]:
        """Return the aliases of the entity."""
        return self.properties["name"][1:] if ("name" in self.properties and len(self.properties["name"]) > 1) else []

    @aliases.setter
    def aliases(self, value: list[str]) -> None:
        """Set the aliases of the entity."""
        self.properties["name"] = (
            [self.name, *value] if "name" in self.properties and len(self.properties["name"]) > 0 else value
        )

    @property
    def names(self) -> list[str]:
        """Return all names of the entity."""
        return self.properties.get("name", [])

    @property
    def description(self) -> str:
        """Return the description of the entity."""
        return self.metadata.get("description", "")

    @description.setter
    def description(self, value: str) -> None:
        """Set the description of the entity."""
        self.metadata["description"] = value

    @property
    def wikipedia_title(self) -> str:
        """Return the Wikipedia title of the entity."""
        return self.metadata.get("wikipedia_title", "")

    @wikipedia_title.setter
    def wikipedia_title(self, value: str) -> None:
        """Set the Wikipedia title of the entity."""
        self.metadata["wikipedia_title"] = value

    @property
    def type(self) -> list[str]:
        """Return the types of the entity."""
        return self.properties[TypeProperties.INSTANCE_OF.value]

    @type.setter
    def type(self, value: list[str]) -> None:
        """Set the types of the entity."""
        self.properties[TypeProperties.INSTANCE_OF.value] = value

    def to_dict(self, minimal_repr: bool = False) -> dict:
        """Convert the WikidataEntity to a dictionary."""
        entity_dict = super().to_dict(minimal_repr=minimal_repr)

        if minimal_repr and "metadata" in entity_dict:
            metadata_dict = entity_dict["metadata"]
            for key in ["description", "wikipedia_title"]:
                if key in metadata_dict and not metadata_dict[key]:
                    del metadata_dict[key]

            if not entity_dict["metadata"]:
                del entity_dict["metadata"]

        return entity_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a WikidataEntity from a dictionary."""
        # TODO(pmyshkov): remove this hacks after the data is re-generated
        if "id" in data:
            data["entity_id"] = data.pop("id")

        if "types" in data:
            data["properties"] = {TypeProperties.INSTANCE_OF.value: data.pop("types")}

        if "name" in data:
            data["properties"]["name"] = [data.pop("name")]

        if not data["properties"]["name"]:
            data["properties"]["name"] = []

        if "aliases" in data:
            data["properties"]["name"].extend(data.pop("aliases"))

        if "metadata" not in data:
            data["metadata"] = {}

        if "description" in data:
            data["metadata"]["description"] = data.pop("description")

        if "wikipedia" in data:
            data["metadata"]["wikipedia_title"] = data.pop("wikipedia")

        return cls(**data)

    @classmethod
    def from_wikidata_record(
        cls,
        entity: dict,
        allowed_properties: Iterable[str] | None = None,
        prohibited_qualifiers: Iterable[str] | None = None,
        include_descriptions: bool = True,
        include_wikipedia_title: bool = True,
    ) -> Self:
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
                    if mainsnak.get("datatype", None) == "quantity":
                        prop_values.append(mainsnak.get("datavalue", {}).get("value", {}).get("amount", None))
                    else:
                        try:
                            prop_values.append(mainsnak.get("datavalue", {}).get("value", {}).get("id"))
                        except AttributeError:
                            prop_values.append(mainsnak.get("datavalue", {}).get("value"))

                if mainsnak.get("snaktype") == "quantity":
                    prop_values.append(mainsnak.get("datavalue", {}).get("value", {}).get("amount", None))

            ent_properties[prop] = prop_values

        wikidata_entity = WikidataEntity(
            entity_id=entity_id,
            properties=ent_properties,
        )

        if include_descriptions:
            wikidata_entity.description = description

        if include_wikipedia_title and "sitelinks" in entity and "enwiki" in entity["sitelinks"]:
            wikidata_entity.wikipedia_title = entity["sitelinks"]["enwiki"]["title"]

        wikidata_entity.properties["name"] = [label, *aliases]

        return wikidata_entity


@dataclass
class ResolvedWikidataEntity(WikidataEntity):
    """A Wikidata entity with resolved properties."""

    @property
    def type(self) -> list[str]:
        """Return the types of the entity."""
        return self.properties["instance of"]

    @type.setter
    def type(self, value: list[str]) -> None:
        """Set the types of the entity."""
        self.properties["instance of"] = value

    @property
    def wikidata_type(self) -> list[str] | None:
        """Return the types of the entity obtained from Wikidata."""
        return self.metadata.get("type", None)

    @wikidata_type.setter
    def wikidata_type(self, value: list[str]) -> None:
        """Set the types of the entity."""
        self.metadata["type"] = value


def _query_entities_via_api(wikidata_ids: Iterable[str], english_only: bool = True) -> dict[str, dict] | None:
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
        response = requests.get(url, params=params, headers={"User-Agent": "KeBAB/WikidataExtractor"})
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
) -> dict[str, dict] | None:
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


def get_id_from_wikidata_claim(claim: dict) -> str | None:
    """Extract the ID from a Wikidata claim."""
    if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
        value = claim["mainsnak"]["datavalue"].get("value", {})
        if isinstance(value, dict) and "id" in value:
            return value["id"]
    return None


def get_entity_label(entity: dict) -> str:
    """Extract the English label from a Wikidata entity."""
    return entity.get("labels", {}).get("en", {}).get("value", "")


def get_entity_aliases(entity: dict) -> set[str]:
    """Extract aliases from a Wikidata entity."""
    aliases = set()
    if "aliases" in entity:
        aliases = {v["value"] for v in entity["aliases"].get("en", []) if "value" in v}
    return aliases


def fetch_properties_via_api(
    start: int = 1,
    end: int = 15_000,
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

    entity_ids = [f"P{i}" for i in range(start, end + 1)]
    entities = query_entities_via_api(entity_ids)
    total_num_additional_aliases = 0
    for prop_id, entity in entities.items():
        if entity:
            assert prop_id == entity["id"]

            prop_record = {
                "label": get_entity_label(entity),
                "description": entity.get("descriptions", {}).get("en", {}).get("value", ""),
                "aliases": get_entity_aliases(entity),
            }

            if not prop_record["label"]:
                continue

            # collect parent properties (P1647)
            parent_properties = entity.get("claims", {}).get("P1647", [])
            parent_properties = [get_id_from_wikidata_claim(claim) for claim in parent_properties]
            parent_properties = [p for p in parent_properties if p is not None]

            wikidata_item_links = entity.get("claims", {}).get(TypeProperties.WIKIDATA_ITEM_OF_THIS_PROPERTY, [])
            if wikidata_item_links:
                wikidata_item_ids = [get_id_from_wikidata_claim(claim) for claim in wikidata_item_links]
                wikidata_item_ids = [id_ for id_ in wikidata_item_ids if id_ is not None]
                if wikidata_item_ids:
                    wikidata_items = query_entities_via_api(wikidata_item_ids)
                    for wikidata_item in wikidata_items.values():
                        label = get_entity_label(wikidata_item)
                        aliases = get_entity_aliases(wikidata_item)
                        logging.debug(
                            f"Found Wikidata item '{label}' with aliases {aliases} for property '{prop_record['label']}'"
                        )
                        num_labels = len(prop_record["aliases"])
                        prop_record["aliases"].add(label)
                        prop_record["aliases"].update(aliases)
                        num_additional_aliases = len(prop_record["aliases"]) - num_labels
                        total_num_additional_aliases += num_additional_aliases
                        if num_additional_aliases > 0:
                            logging.debug(
                                f"Added {num_additional_aliases} new aliases from Wikidata item '{label}' to property '{prop_record['label']}'"
                            )

            prop_record["subproperty_of"] = parent_properties
            prop_record["aliases"] = list(prop_record["aliases"]) if prop_record["aliases"] else []

            properties[prop_id] = prop_record
        else:
            error_count += 1

    if error_count:
        logging.warning(f"Encountered {error_count} errors while querying Wikidata")

    logging.info(f"Finished processing {end - start + 1} records, found {len(properties)} properties")
    logging.info(
        f"Total number of additional aliases added from property 'Wikidata item of this property': {total_num_additional_aliases}"
    )
    for k, v in properties.items():
        logging.debug(f"{k}: {v}")

    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(properties, f)


def extract_wikidata_entities_from_dump(
    wikidata_json_dump_path: pathlib.Path = pathlib.Path.cwd() / "latest-all.json",
    properties: Iterable[str] | None = None,
    input_entity_predicate: Callable[[dict], bool] | None = None,
    output_entity_predicate: Callable[[WikidataEntity], bool] | None = None,
    include_descriptions: bool = False,
    include_wikipedia_title: bool = False,
) -> Iterable[WikidataEntity]:
    """Extract all entities that match the predicate and their requested properties from the Wikidata JSON dump."""
    logging.info(f"Reading {wikidata_json_dump_path}...")

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
                logging.info(f"Read {processed_count:,} records")

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

                wikidata_entity = WikidataEntity.from_wikidata_record(
                    entity,
                    allowed_properties=properties,
                    include_descriptions=include_descriptions,
                    include_wikipedia_title=include_wikipedia_title,
                )

                # check if the input predicate applies
                if not output_entity_predicate(wikidata_entity):
                    continue

                yield wikidata_entity

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
    """Load the hierarchy of Wikidata types.

    Returns:
        A tuple containing the type hierarchy graph and a map from all type IDs to their nodes.
    """
    logging.info("Loading the hierarchy of Wikidata types")
    graph = {}
    type_id_to_node_map = {}

    with open(type_hierarchy_path, encoding="utf-8") as f:
        for line in f:
            node = json.loads(line.strip())
            graph[node["id"]] = node
            for node_id in node["merged_ids"]:
                type_id_to_node_map[node_id] = node

            for node_id in node["redirect_from_ids"]:
                type_id_to_node_map[node_id] = node

    logging.info(f"Type hierarchy contains {len(graph):,d} nodes")

    return graph, type_id_to_node_map


def load_wikidata_entities(wikidata_entities_path: pathlib.Path) -> Iterable[WikidataEntity]:
    """Load the list of wikidata entities extracted from Wikidata."""
    count = 0
    with open(wikidata_entities_path, encoding="utf-8") as f:
        for line in f:
            entity = WikidataEntity.from_json(line.strip())
            count += 1
            yield entity

    logging.info(f"Read {count:,d} entities")


def collect_all_subtypes(graph: dict[str, dict], type_id_to_node: dict[str, dict], type_id: str) -> set[str]:
    """Collect all subtypes of the given type from the type hierarchy."""
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


def collect_wikidata_entities(
    ids_to_include: set[str],
    wikidata_entities_path: pathlib.Path,
    query_api: bool = False,
    append_to_file: bool = False,
    include_properties: bool = False,
    include_descriptions: bool = False,
) -> dict[str, WikidataEntity]:
    """Collect Wikidata entities for the specified IDs from a file and query API for any missing entities ."""
    logging.info(f"Loading Wikidata entities from {wikidata_entities_path}")

    required_ids = set(ids_to_include)
    entities = {}
    with open(wikidata_entities_path, encoding="utf-8") as f:
        for line in f:
            entity = WikidataEntity.from_json(line.strip())

            if entity.entity_id not in required_ids:
                continue

            entities[entity.entity_id] = entity
            required_ids.remove(entity.entity_id)

    if required_ids:
        if query_api:
            logging.info(f"Querying {len(required_ids):,} entities via the Wikidata API")

            q_entities = query_entities_via_api(required_ids)

            if q_entities is None:
                logging.error("Failed to query entities via the Wikidata API.")
                return entities

            q_entities = {
                k: WikidataEntity.from_wikidata_record(
                    e, allowed_properties=[] if include_properties else None, include_descriptions=include_descriptions
                )
                for k, e in q_entities.items()
                if e is not None
            }

            if append_to_file:
                with open(wikidata_entities_path, mode="a", encoding="utf-8") as f:
                    for entity in q_entities.values():
                        f.write(json.dumps(entity, ensure_ascii=False) + "\n")
                logging.info(f"Appended {len(q_entities):,} entities to {wikidata_entities_path}")

            entities.update(q_entities)
        else:
            logging.warning(f"Failed to find {len(required_ids):,} entities")

    logging.info(f"Found {len(entities):,} Wikidata entities.")

    return entities


def get_wikipedia_titles(wikidata_ids: list[str]) -> Iterable[tuple[str, str]]:
    """Get Wikipedia titles for a list of Wikidata IDs."""

    def get_wikipedia_titles_batch(wikidata_ids: list[str]) -> dict[str, str]:
        """Get Wikipedia titles for a list of Wikidata IDs."""
        url = "https://www.wikidata.org/w/api.php"
        headers = {"User-Agent": "MyWikipediaBot/1.0 (myemail@example.com)"}
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(wikidata_ids),
            "props": "sitelinks",
            "sitefilter": "enwiki",
        }

        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        titles = {}
        for wikidata_id, entity in data.get("entities", {}).items():
            if "sitelinks" in entity and "enwiki" in entity["sitelinks"]:
                titles[wikidata_id] = entity["sitelinks"]["enwiki"]["title"]
            else:
                titles[wikidata_id] = None

        return titles

    def chunked_list(lst: list, n: int) -> Iterable[list]:
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for chunk in chunked_list(wikidata_ids, 50):
        try:
            results = get_wikipedia_titles_batch(chunk)
        except ValueError as e:
            logging.error(f"Error getting Wikipedia titles: {e}")
            continue

        yield from results.items()


def get_wikipedia_intros(wikidata_ids: list[str]) -> Iterable[tuple[str, str]]:
    """Get Wikipedia intros for a list of Wikidata IDs."""

    def get_wikipedia_intro_batch(titles: list[str]) -> dict[str, str]:
        """Get Wikipedia intros for a list of Wikidata IDs."""
        wiki_url = "https://en.wikipedia.org/w/api.php"
        headers = {"User-Agent": "MyWikipediaBot/1.0 (myemail@example.com)"}
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(titles),
            "prop": "extracts",
            "explaintext": True,
            "exintro": True,  # Retrieve only the introduction
        }

        response = requests.get(wiki_url, params=params, headers=headers)
        data = response.json()

        results = {}
        pages = data["query"]["pages"]
        for page_info in pages.values():
            results[page_info["title"]] = page_info.get("extract", "No content found.")

        return results

    def chunked_list(lst: list, n: int) -> Iterable[list]:
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    done = 0
    skipped = 0

    for chunk in chunked_list(wikidata_ids, 10):
        done += len(chunk)

        try:
            titles = get_wikipedia_titles(chunk)
            titles = {k: v for k, v in titles if v is not None}
            skipped += len(chunk) - len(titles)
        except ValueError as e:
            logging.error(f"Error getting Wikipedia titles: {e}")
            skipped += len(chunk)
            continue

        try:
            texts = get_wikipedia_intro_batch(list(titles.values()))
        except ValueError as e:
            logging.error(f"Error getting Wikipedia intros: {e}")
            skipped += len(titles)
            continue

        with_pages = 0
        for wikidata_id, title in titles.items():
            text = texts.get(title, "")
            if text:
                with_pages += 1
                yield wikidata_id, text

        skipped += len(titles) - with_pages

        if done % 1000 == 0:
            logging.info(
                f"Processed {done} entities, skipped {skipped} ({skipped / done:.2f}) without Wikipedia titles/pages"
            )
