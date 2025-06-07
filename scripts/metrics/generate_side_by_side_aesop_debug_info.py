from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def to_list_of_strings(serialized_list: str) -> list[str]:
    """Convert a serialized list of strings to a list of strings."""
    return [item.strip().strip("'") for item in serialized_list.strip("[]").split(",") if item.strip().strip("'")]


def to_list_of_float(serialized_list: str) -> list[float]:
    """Convert a serialized list of floats to a list of floats."""
    return [float(item.strip()) for item in serialized_list.strip("[]").split(",") if item.strip()]


def empty_row(columns: list[str]) -> pd.DataFrame:
    """Create an empty DataFrame with specified columns."""
    return pd.DataFrame.from_records([dict.fromkeys(columns, "")])


def load_debug_info(filename: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load debug info from a CSV file and return a dictionary of DataFrames."""
    debug_info = pd.read_csv(
        filename,
        converters={
            "property_id": str,
            "gt_values": to_list_of_strings,
            "pred_values": to_list_of_strings,
            "matched_scores": to_list_of_float,
            "unmatched_gt_values": to_list_of_strings,
            "unmatched_pred_values": to_list_of_strings,
        },
    )
    entity_rows = []
    gt_names_to_df = {}
    gt_names = None
    pred_values_idx = 0
    for row in debug_info.itertuples(index=False):
        if row.property_id == "" and gt_names is not None:  # type: ignore
            entity_df = pd.DataFrame.from_records(sorted(entity_rows, key=lambda x: x["property_id"]))
            entity_df["all_gt_names"] = str(gt_names)
            gt_names_to_df[gt_names] = entity_df
            entity_rows = []
            gt_names = None
        elif row.property_id == "name":  # type: ignore
            gt_names = tuple(row.gt_values + row.unmatched_gt_values)  # type: ignore
            if not gt_names:
                # reached unmatched pred_values
                gt_names = None
                break
            entity_rows.append(row._asdict())  # type: ignore
        else:
            entity_rows.append(row._asdict())  # type: ignore
        pred_values_idx += 1
    if gt_names is not None and entity_rows:
        entity_df = pd.DataFrame.from_records(sorted(entity_rows, key=lambda x: x["property_id"]))
        entity_df["all_gt_names"] = gt_names
        gt_names_to_df[gt_names] = entity_df
    unmatched_pred_df = debug_info.iloc[pred_values_idx:]
    unmatched_pred_df["all_gt_names"] = str([])
    return gt_names_to_df, unmatched_pred_df  # type: ignore


def generate_side_by_side_comparison(
    filename1: str, filename2: str, output_filename: str, suffixes: tuple[str, str] = ("_1", "_2")
) -> None:
    """Generate a side-by-side comparison of two debug info CSV files."""
    gt_names_to_df1, unmatched_pred_df1 = load_debug_info(filename1)
    gt_names_to_df2, unmatched_pred_df2 = load_debug_info(filename2)

    dfs = []
    columns = ["gt_values", "pred_values", "matched_scores", "unmatched_gt_values", "unmatched_pred_values"]
    merged_columns = ["property_id", "all_gt_names"] + [col + suffix for col in columns for suffix in suffixes]
    key_intersection = set(gt_names_to_df1.keys()).intersection(set(gt_names_to_df2.keys()))
    suffixes = ("_phi4", "_phi3")
    for key in key_intersection:
        df1 = gt_names_to_df1[key]
        df2 = gt_names_to_df2[key]
        dfs.append(df1.merge(df2, on=["property_id", "all_gt_names"], suffixes=suffixes)[merged_columns])
        dfs.append(empty_row(merged_columns))
    unmatched_pred_df1 = unmatched_pred_df1.rename(mapper={col: col + suffixes[0] for col in columns}, axis=1)
    unmatched_pred_df2 = unmatched_pred_df2.rename(mapper={col: col + suffixes[1] for col in columns}, axis=1)
    for column in columns:
        unmatched_pred_df1[column + suffixes[1]] = ""
        unmatched_pred_df2[column + suffixes[0]] = ""
    dfs.append(unmatched_pred_df1)
    dfs.append(unmatched_pred_df2)
    return pd.concat(dfs, ignore_index=True).to_csv(output_filename, index=False)


def main():
    """Entry point for the script to generate side-by-side comparison of debug info CSV files."""
    parser = ArgumentParser(description="Generate side-by-side comparison of debug info CSV files.")
    parser.add_argument("dir1", type=Path, help="Directory containing the first set of debug info CSV files.")
    parser.add_argument("dir2", type=Path, help="Directory containing the second set of debug info CSV files.")
    parser.add_argument("output_dir", type=Path, help="Directory to save the comparison CSV files.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for file1 in args.dir1.glob("*.csv"):
        file2 = args.dir2 / file1.name
        if file2.exists():
            output_filename = args.output_dir / file1.name.replace(".csv", "_comparison.csv")
            generate_side_by_side_comparison(file1, file2, output_filename, suffixes=("_phi4", "_phi3"))
            print(f"Generated comparison for {file1.name} and {file2.name} at {output_filename}.")
        else:
            print(f"File {file2} does not exist. Skipping comparison.")


if __name__ == "__main__":
    main()
