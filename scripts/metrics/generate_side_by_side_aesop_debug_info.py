from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from kebab.tasks.metrics.extraction.aesop.metric_helpers import UNDEFINED_ID, UNDEFINED_VALUE, UNDEFINED_SCORE


def empty_row(columns: list[str]) -> pd.DataFrame:
    """Create an empty DataFrame with specified columns."""
    return pd.DataFrame.from_records([dict.fromkeys(columns, "")])


def drop_last_columns(df: pd.DataFrame, num_cols_to_drop: int) -> pd.DataFrame:
    """Drop the last `num_cols_to_drop` columns from a DataFrame."""
    if num_cols_to_drop <= 0:
        return df
    return df.drop(columns=df.columns[-num_cols_to_drop:].tolist(), axis=1)


def split_df_by_document_id(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a DataFrame into multiple DataFrames by document_id."""
    split_dfs = {}
    for document_id, group in df.groupby("document_id"):
        split_dfs[document_id] = group.reset_index(drop=True)
    return split_dfs


def split_by_gt_entity(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a DataFrame into multiple DataFrames by entity_id."""
    split_dfs = {}
    for entity_id, group in df.groupby("gt_entity_id"):
        split_dfs[entity_id] = group.reset_index(drop=True)
    return split_dfs


def split_by_pred_entity(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a DataFrame into multiple DataFrames by predicted entity_id."""
    split_dfs = {}
    for _, group in df.groupby("pred_entity_id"):
        split_dfs[group["name"].numpy()[0]] = group.reset_index(drop=True)
    return split_dfs


def add_suffix_excluding_columns(
    df: pd.DataFrame, suffix: str, excluded_columns: Iterable[str] | None = None
) -> pd.DataFrame:
    """Add a suffix to all columns except the excluded ones."""
    excluded_columns_set = set() if excluded_columns is None else set(excluded_columns)
    return df.rename(mapper=lambda x: x + suffix if x not in excluded_columns_set else x, axis=1)


def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop specified columns from a DataFrame."""
    return df.drop(columns=columns, errors="ignore")


def merge_entity_dfs(df_left: pd.DataFrame, df_right: pd.DataFrame, suffixes: tuple[str, str]) -> pd.DataFrame:
    """Merge two DataFrames containing entity information."""
    excluded_columns = {"document_id", "original_text", "property_id", "gt_entity_id", "gt_values"}
    df_left = add_suffix_excluding_columns(df_left, suffixes[0], excluded_columns)
    df_right = add_suffix_excluding_columns(df_right, suffixes[1], excluded_columns)
    matched = df_left[df_left["property_id"] != UNDEFINED_ID].merge(
        df_right[df_right["property_id"] != UNDEFINED_ID],
        on=list(excluded_columns),
        suffixes=suffixes,
    )
    unmatched_right = df_right[df_right["property_id"] == UNDEFINED_ID].drop(columns=list(excluded_columns))
    unmatched_left = df_left[df_left["property_id"] == UNDEFINED_ID]
    unmatched = unmatched_left.merge(
        unmatched_right,
        left_on=[f"original_pred_property_id{suffixes[0]}", f"pred_values{suffixes[0]}"],
        right_on=[f"original_pred_property_id{suffixes[1]}", f"pred_values{suffixes[1]}"],
        how="outer",
    )
    result = pd.concat([matched, unmatched], ignore_index=True)
    return result


def split_into_gt_and_unmatched_pred(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into two rowwise: for ground truth entities and unmatched predicted entities."""
    return pd.DataFrame(df[df["gt_entity_id"] != UNDEFINED_ID]), pd.DataFrame(df[df["gt_entity_id"] == UNDEFINED_ID])


def merge_document_dfs(document_dfs: list[pd.DataFrame], suffixes: tuple[str, str] | None = None) -> pd.DataFrame:
    """Merge two DataFrames containing debug information for the same document."""
    gt_left, unmatched_pred_left = split_into_gt_and_unmatched_pred(document_dfs[0])
    gt_right, unmatched_pred_right = split_into_gt_and_unmatched_pred(document_dfs[1])
    common_columns = ["document_id", "original_text", "gt_entity_id", "property_id", "gt_values"]

    if suffixes is None:
        suffixes = ("_1", "_2")

    columns_left = list(add_suffix_excluding_columns(document_dfs[0], suffixes[0], set(common_columns)).columns)
    columns_right = list(add_suffix_excluding_columns(document_dfs[1], suffixes[1], set(common_columns)).columns)
    gt_id_to_entity_left = split_by_gt_entity(gt_left)
    gt_id_to_entity_right = split_by_gt_entity(gt_right)

    all_gt_ids = gt_id_to_entity_left.keys()

    merged_gt = [
        merge_entity_dfs(gt_id_to_entity_left[gt_entity_id], gt_id_to_entity_right[gt_entity_id], suffixes)
        for gt_entity_id in all_gt_ids
    ]

    merged_gt_df = None
    if merged_gt:
        merged_gt_df = pd.concat(merged_gt, ignore_index=True)

    name_to_entity_left = split_by_pred_entity(unmatched_pred_left)
    name_to_entity_right = split_by_pred_entity(unmatched_pred_right)

    matched_names = set(name_to_entity_left.keys()) & set(name_to_entity_right.keys())

    merged_unmatched_pred = [
        merge_entity_dfs(name_to_entity_left[name], name_to_entity_right[name], suffixes) for name in matched_names
    ]

    merged_unmatched_pred_df = None
    if merged_unmatched_pred:
        merged_unmatched_pred_df = pd.concat(merged_unmatched_pred)

    left_unmatched_names = [entity for name, entity in name_to_entity_left.items() if name not in matched_names]
    right_unmatched_names = [entity for name, entity in name_to_entity_right.items() if name not in matched_names]
    if left_unmatched_names:
        unmatched_pred_left_remainder = pd.concat(left_unmatched_names, ignore_index=True)
        unmatched_pred_left_remainder = add_suffix_excluding_columns(
            unmatched_pred_left_remainder, suffixes[0], common_columns
        )
    else:
        unmatched_pred_left_remainder = empty_row(columns_left)

    if right_unmatched_names:
        unmatched_pred_right_remainder = pd.concat(right_unmatched_names, ignore_index=True)
        unmatched_pred_right_remainder = add_suffix_excluding_columns(
            unmatched_pred_right_remainder, suffixes[1], common_columns
        )
    else:
        unmatched_pred_right_remainder = empty_row(columns_right)

    unmatched_pred_right_remainder = drop_columns(unmatched_pred_right_remainder, common_columns)

    unmatched_df = pd.concat([unmatched_pred_left_remainder, unmatched_pred_right_remainder], axis=1)

    result = pd.concat(
        [df for df in [merged_gt_df, merged_unmatched_pred_df, unmatched_df] if df is not None], ignore_index=True
    )
    return result


def generate_side_by_side_comparison(
    filename1: str, filename2: str, output_filename: str, suffixes: tuple[str, str] | None = None
) -> None:
    """Generate a side-by-side comparison of two debug info XLSX files."""
    num_cols_to_drop = 5  # Assuming the last 5 columns are metrics
    dfs = [drop_last_columns(pd.read_excel(filename), num_cols_to_drop) for filename in [filename1, filename2]]
    doc_id_to_dfs = defaultdict(list)
    for df in dfs:
        split_dfs = split_df_by_document_id(df)
        for document_id, split_df in split_dfs.items():
            doc_id_to_dfs[document_id].append(split_df)
    merged_doc_dfs = []
    for document_dfs in doc_id_to_dfs.values():
        merged_doc_df = merge_document_dfs(document_dfs, suffixes)
        merged_doc_dfs.append(merged_doc_df)
    result = pd.concat(merged_doc_dfs, ignore_index=True)
    result.to_excel(output_filename, index=False)


def main():
    """Entry point for the script to generate side-by-side comparison of debug info XLSX files."""
    parser = ArgumentParser(description="Generate side-by-side comparison of debug info XLSX files.")
    parser.add_argument("dir1", type=Path, help="Directory containing sets of debug info XSLX files.")
    parser.add_argument("dir2", type=Path, help="Directory containing sets of debug info XLSX files.")
    parser.add_argument("output_dir", type=Path, help="Directory to save the comparison XLSX files.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dir_ in [args.dir1, args.dir2]:
        if not dir_.is_dir():
            raise ValueError(f"Provided path {dir_} is not a directory.")
    for file1 in args.dir1.glob("*.xlsx"):
        file2 = args.dir2 / file1.name
        if not file2.is_file():
            print(f"File {file2} does not exist. Skipping.")
            continue
        output_filename = args.output_dir / file1.name.replace(".xlsx", "_comparison.xlsx")
        generate_side_by_side_comparison(file1, file2, output_filename, suffixes=("_1", "_2"))
        print(f"Comparison file for {file1.name} saved to {args.output_dir}")

if __name__ == "__main__":
    main()
