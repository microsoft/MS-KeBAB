import argparse
import json
import os
from typing import List, Set, Tuple

parser = argparse.ArgumentParser("get_metaqa_details")
parser.add_argument(
    "--metaqa-path", type=str, required=True, help="path to MetaQA root directory"
)
parser.add_argument(
    "--output-path",
    type=str,
    required=True,
    help="path to directory with generated outputs",
)

one_hop_steps = {
    "movie_to_year": ["release_year"],
    "movie_to_director": ["directed_by"],
    "movie_to_actor": ["starred_actors"],
    "movie_to_tags": ["has_tags"],
    "movie_to_writer": ["written_by"],
    "movie_to_language": ["in_language"],
    "movie_to_imdbvotes": ["has_imdb_votes"],
    "movie_to_genre": ["has_genre"],
    "movie_to_imdbrating": ["has_imdb_rating"],
    "tag_to_movie": ["movie_has_tags"],
    "actor_to_movie": ["movie_starred_actors"],
    "director_to_movie": ["movie_directed_by"],
    "writer_to_movie": ["movie_written_by"],
}

two_hop_steps = {
    "actor_to_movie_to_director": ["movie_starred_actors", "directed_by"],
    "writer_to_movie_to_year": ["movie_written_by", "release_year"],
    "actor_to_movie_to_language": ["movie_starred_actors", "in_language"],
    "director_to_movie_to_actor": ["movie_directed_by", "starred_actors"],
    "actor_to_movie_to_year": ["movie_starred_actors", "release_year"],
    "movie_to_director_to_movie": ["directed_by", "movie_directed_by"],
    "director_to_movie_to_writer": ["movie_directed_by", "written_by"],
    "director_to_movie_to_genre": ["movie_directed_by", "has_genre"],
    "actor_to_movie_to_genre": ["movie_starred_actors", "has_genre"],
    "writer_to_movie_to_language": ["movie_written_by", "in_language"],
    "writer_to_movie_to_writer": ["movie_written_by", "written_by"],
    "writer_to_movie_to_director": ["movie_written_by", "directed_by"],
    "director_to_movie_to_language": ["movie_directed_by", "in_language"],
    "actor_to_movie_to_actor": ["movie_starred_actors", "starred_actors"],
    "writer_to_movie_to_genre": ["movie_written_by", "has_genre"],
    "movie_to_writer_to_movie": ["written_by", "movie_written_by"],
    "actor_to_movie_to_writer": ["movie_starred_actors", "written_by"],
    "writer_to_movie_to_actor": ["movie_written_by", "starred_actors"],
    "movie_to_actor_to_movie": ["starred_actors", "movie_starred_actors"],
    "director_to_movie_to_director": ["movie_directed_by", "movie_directed_by"],
    "director_to_movie_to_year": ["movie_directed_by", "release_year"],
}

three_hop_steps = {
    "movie_to_director_to_movie_to_genre": [
        "directed_by",
        "movie_directed_by",
        "has_genre",
    ],
    "movie_to_actor_to_movie_to_genre": [
        "starred_actors",
        "movie_starred_actors",
        "has_genre",
    ],
    "movie_to_writer_to_movie_to_year": [
        "written_by",
        "movie_written_by",
        "release_year",
    ],
    "movie_to_director_to_movie_to_year": [
        "directed_by",
        "movie_directed_by",
        "release_year",
    ],
    "movie_to_actor_to_movie_to_year": [
        "starred_actors",
        "movie_starred_actors",
        "release_year",
    ],
    "movie_to_actor_to_movie_to_director": [
        "starred_actors",
        "movie_starred_actors",
        "directed_by",
    ],
    "movie_to_director_to_movie_to_actor": [
        "directed_by",
        "movie_directed_by",
        "starred_actors",
    ],
    "movie_to_writer_to_movie_to_actor": [
        "written_by",
        "movie_written_by",
        "starred_actors",
    ],
    "movie_to_actor_to_movie_to_writer": [
        "starred_actors",
        "movie_starred_actors",
        "written_by",
    ],
    "movie_to_actor_to_movie_to_language": [
        "starred_actors",
        "movie_starred_actors",
        "in_language",
    ],
    "movie_to_writer_to_movie_to_genre": [
        "written_by",
        "movie_written_by",
        "has_genre",
    ],
    "movie_to_writer_to_movie_to_language": [
        "written_by",
        "movie_written_by",
        "in_language",
    ],
    "movie_to_director_to_movie_to_language": [
        "directed_by",
        "movie_directed_by",
        "in_language",
    ],
    "movie_to_writer_to_movie_to_director": [
        "written_by",
        "movie_written_by",
        "directed_by",
    ],
    "movie_to_director_to_movie_to_writer": [
        "directed_by",
        "movie_directed_by",
        "written_by",
    ],
}


def reverse_steps(steps: List[str]) -> List[str]:
    def reverse_relation_name(name: str) -> str:
        if name.startswith("movie_"):
            return name[6:]
        return "movie_" + name

    return list(reversed([reverse_relation_name(step) for step in steps]))


class EntityIdCounter:
    def __init__(self):
        self._next_id = 0

    def get_next_id(self) -> int:
        return self._next_id

    def increment_id(self):
        self._next_id += 1


names_to_entity_ids = {}
id_counter = EntityIdCounter()


def add_value_to_dict_of_list(dictionary: dict, property, value):
    if property not in dict:
        dictionary[property] = []
    dictionary[property].append(value)


def parse_line(line: str, prev_entity_name: str, kb_list: list) -> str:
    fact = line.split("|")
    entity_name_a, relation, entity_name_b = fact[0], fact[1], fact[2]
    if entity_name_a != prev_entity_name or entity_name_a not in names_to_entity_ids:
        kb_list.append({"entity_id": id_counter.get_next_id(), "name": entity_name_a})
        add_value_to_dict_of_list(
            names_to_entity_ids, entity_name_a, id_counter.get_next_id()
        )
        id_counter.increment_id()
    if entity_name_b not in names_to_entity_ids:
        kb_list.append({"entity_id": id_counter.get_next_id(), "name": entity_name_b})
        add_value_to_dict_of_list(
            names_to_entity_ids, entity_name_b, id_counter.get_next_id()
        )
        id_counter.increment_id()
    entity_id = names_to_entity_ids[entity_name_a][-1]
    entity_other_id = names_to_entity_ids[entity_name_b][0]
    entity = kb_list[entity_id]
    entity_other = kb_list[entity_other_id]
    reverse_relation = f"movie_{relation}"
    add_value_to_dict_of_list(entity, relation, entity_other_id)
    add_value_to_dict_of_list(entity_other, reverse_relation, entity_id)
    return entity_name_a


def create_json_kb(kb_raw_path: str, kb_json_output_path: str) -> list:
    kb_list = []
    prev_entity_name = ""
    with open(kb_raw_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            prev_entity_name = parse_line(line.strip(), prev_entity_name, kb_list)
    with open(kb_json_output_path, "w", encoding="utf-8") as f:
        json.dump(kb_list, f)
    return kb_list


def get_ids_or_empty(dictionary: dict, key) -> list:
    return dictionary[key] if key in dictionary else []


def perform_search(
    base_query_entity_ids: List[int], hop_types: List[str]
) -> Tuple[Set[int], List[int]]:
    all_relevant = set(base_query_entity_ids)
    to_expand = []
    next_to_expand = base_query_entity_ids.copy()
    for hop_type in hop_types:
        to_expand = next_to_expand
        next_to_expand = []
        while to_expand:
            entity_expanded_id = to_expand.pop()
            entity_expanded = kb_list[entity_expanded_id]
            if hop_type not in entity_expanded:
                continue
            for neighbour in entity_expanded[hop_type]:
                if neighbour not in next_to_expand:
                    all_relevant.add(neighbour)
                    next_to_expand.append(neighbour)
    return (all_relevant, next_to_expand)


def get_all_relevant_entities(
    base_query_entity_id: int, hop_types: List[str]
) -> Tuple[Set[int], List[int]]:
    all_visited, all_reached = perform_search([base_query_entity_id], hop_types)
    all_visited_reversed, _ = perform_search(all_reached, reverse_steps(hop_types))
    return (all_visited.intersection(all_visited_reversed), all_reached)


def get_questions_and_types(hops_count: int, split: str) -> zip:
    base_path = f"{hops_count}-hop"
    query_types_file = os.path.join(base_path, f"qa_{split}_qtype.txt")
    with open(query_types_file, "r", encoding="utf-8") as f:
        query_types = [line.strip() for line in f.readlines()]
    queries_file = os.path.join(base_path, "vanilla", f"qa_{split}.txt")
    base_query_entities = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            query = line.strip()
            open_bracket_idx = query.find("[")
            close_bracket_idx = query.find("]")
            relevant_entity_name = query[open_bracket_idx + 1 : close_bracket_idx]
            base_query_entities.append(names_to_entity_ids[relevant_entity_name][0])
    return zip(base_query_entities, query_types)


def compute_ground_truth_entities_1_hop_single(
    base_query_entity_id: int, query_type: str
) -> Tuple[Set[int], List[int]]:
    return get_all_relevant_entities(base_query_entity_id, one_hop_steps[query_type])


def compute_ground_truth_entities_2_hop_single(
    base_query_entity_id: int, query_type: str
) -> Tuple[Set[int], List[int]]:
    return get_all_relevant_entities(base_query_entity_id, two_hop_steps[query_type])


def compute_ground_truth_entities_3_hop_single(
    base_query_entity_id: int, query_type: str
) -> Tuple[Set[int], List[int]]:
    return get_all_relevant_entities(base_query_entity_id, three_hop_steps[query_type])


def compute_all_ground_truth_one_hop(split: str) -> List[Tuple[Set[int], List[int]]]:
    return [
        compute_ground_truth_entities_1_hop_single(entity_id, query_type)
        for entity_id, query_type in get_questions_and_types(1, split)
    ]


def compute_all_ground_truth_two_hop(split: str) -> List[Tuple[Set[int], List[int]]]:
    return [
        compute_ground_truth_entities_2_hop_single(entity_id, query_type)
        for entity_id, query_type in get_questions_and_types(2, split)
    ]


def compute_all_ground_truth_three_hop(split: str) -> List[Tuple[Set[int], List[int]]]:
    return [
        compute_ground_truth_entities_3_hop_single(entity_id, query_type)
        for entity_id, query_type in get_questions_and_types(3, split)
    ]


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    kb_txt_input_path = os.path.join(args.metaqa_path, "kb.txt")
    kb_json_output_path = os.path.join(args.output_path, "kb.json")
    kb_list = create_json_kb(
        kb_txt_input_path,
        kb_json_output_path,
    )
    print(f"KB created and saved to {kb_json_output_path}")
    hop_computation_functions = [
        compute_all_ground_truth_one_hop,
        compute_all_ground_truth_two_hop,
        compute_all_ground_truth_three_hop,
    ]
    for hop_idx, hop_function in enumerate(hop_computation_functions):
        hops_count = hop_idx + 1
        for split in ["train", "dev", "test"]:
            print(f"{split} split, {hops_count} hops")
            output_dir_for_split = os.path.join(
                args.output_path, f"hops_{hops_count}", split
            )
            os.makedirs(output_dir_for_split, exist_ok=True)
            ground_truth = hop_function(split)
            with (
                open(
                    os.path.join(output_dir_for_split, "all_relevant.txt"),
                    "w",
                    encoding="utf-8",
                ) as f_all,
                open(
                    os.path.join(output_dir_for_split, "final_answer.txt"),
                    "w",
                    encoding="utf-8",
                ) as f_final,
            ):
                for all_relevant, final_answer in ground_truth:
                    all_relevant_ids = [str(entity_id) for entity_id in all_relevant]
                    final_ids = [str(entity_id) for entity_id in final_answer]
                    f_all.write("|".join(all_relevant_ids))
                    f_all.write("\n")
                    f_final.write("|".join(final_ids))
                    f_final.write("\n")
