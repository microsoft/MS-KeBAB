# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# pyright: reportReturnType=false, reportOptionalMemberAccess=false
# ruff: noqa: TRY400, S113, PLR2004, S101, RET505, B008, A001, PLW2901, PLR1714, ARG005
"""Utilities for extracting from Wikidata."""

from __future__ import annotations

import json
import logging
import pathlib
import time
from collections.abc import Callable, Iterable

import requests
from tqdm import tqdm


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
        entities = response.json()["entities"]
        assert len(entities) == len(wikidata_ids)

        # NOTE: the below fails due to redirects
        # for entity_id, entity in entities.items():
        #     assert entity_id == entity["id"]
        #     assert entity["id"] in wikidata_ids
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


def convert_to_simple_entity(
    entity: dict,
    properties: Iterable[str] | None = None,
    prohibited_qualifiers: Iterable[str] | None = None,
) -> dict:
    """Convert a Wikidata entity to a simpler representation."""
    properties = set(properties) if properties else None

    entity_id = entity["id"]
    label = entity.get("labels", {}).get("en", {}).get("value", "")

    description = entity.get("descriptions", {}).get("en", {}).get("value", "")
    aliases = [v["value"] for v in entity.get("aliases", {}).get("en", {})]

    # extract properties
    claims = entity.get("claims", {})
    keys = set(claims.keys())
    if properties is not None:
        keys &= properties

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

    simple_entity = {
        "id": entity_id,
        "name": label,
        "description": description,
        "aliases": aliases,
        "properties": ent_properties,
    }

    return simple_entity


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


def extract_entities_from_dump(
    path_to_dump_json: pathlib.Path = pathlib.Path.cwd() / "latest-all.json",
    output_path: pathlib.Path = pathlib.Path.cwd() / "wikidata_extracted_entities.jsonl",
    properties: Iterable[str] | None = None,
    input_entity_predicate: Callable[[dict], bool] | None = None,
    output_entity_predicate: Callable[[dict], bool] | None = None,
) -> pathlib.Path:
    """Extract all items that match the predicate and their requested properties from the Wikidata JSON dump."""
    logging.info(f"Extracting entities from {path_to_dump_json}...")

    processed_count = 0
    error_count = 0
    skipped_count = 0

    properties = set(properties) if properties else None
    input_entity_predicate = input_entity_predicate or (lambda x: True)
    output_entity_predicate = output_entity_predicate or (lambda x: True)

    with open(output_path, mode="w", encoding="utf-8") as f_out, open(path_to_dump_json, encoding="utf-8") as f_in:
        for line in f_in:
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

                ent = convert_to_simple_entity(entity, properties=properties)

                # check if the input predicate applies
                if not output_entity_predicate(ent):
                    continue

                f_out.write(json.dumps(ent) + "\n")

            except json.JSONDecodeError:
                logging.warning(f"Error parsing line: {line}")
                error_count += 1

    if error_count:
        logging.warning(f"Encountered {error_count} JSON parsing errors")

    if skipped_count:
        logging.warning(f"Skipped {skipped_count} entities without labels or type")

    return output_path


def load_properties(path: pathlib.Path) -> dict[str, dict]:
    """Load (previously extracted) Wikidata properties from a JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
    except json.JSONDecodeError:
        logging.error(f"Error while decoding JSON: {path}")
