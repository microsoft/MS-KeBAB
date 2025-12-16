# A command-line utility that will read all tsv files
# whose names start with "linking_predictions" in a folder
# and merge them into a single predictions file.
import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def safe_merge(df1: pd.DataFrame, df2: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Merge two dataframes on specified columns, with error handling for missing keys."""
    try:
        return df1.merge(df2, on=on)
    except KeyError:
        print(
            "KeyError here means that the join columns are not consistent across files.  Try extending system_columns with the missing columns."
        )
        missing_left = [c for c in on if c not in df1.columns and c not in (df1.index.names or [])]
        missing_right = [c for c in on if c not in df2.columns and c not in (df2.index.names or [])]
        print("Missing on left:", missing_left)
        print("Missing on right:", missing_right)
        raise


def main():
    """Entry point for the script to generate merged linking predictions TSV files."""
    parser = argparse.ArgumentParser(
        description="""
    This will merge all linking_predictions*.tsv files from the specified folders.
    The merged file will be named 'merged_linking_predictions.tsv'.
    Each system's columns will be suffixed with the folder name, excluding the common prefix.
    If a tsv file has a suffix in its name (after 'linking_predictions'), that suffix will also be added.
    All metrics*.json files will also be merged into 'merged_metrics.tsv' similarly.
    """
    )
    parser.add_argument("folder", nargs="+", type=Path, help="folder to merge")
    args = parser.parse_args()
    folders = args.folder
    # Expand wildcards
    folders = [p for folder in folders for p in glob.glob(str(folder)) if Path(p).is_dir()]
    # Find the common prefix of all folder names
    common_prefix = Path(os.path.commonprefix(folders))
    merged_data_frame = None
    merged_metrics = None
    predicted_labels = {}
    # We hope that join_columns are the same across dataframes and that a combined value of these columns uniquely identifies the row.
    join_columns = [
        "left",
        "right",
        "entity_type",
        "overlap_props",
        "prop_overlap_num",
        "prop_pattern",
        "name_overlap",
        "label",
        "left_entity_id",
        "right_entity_id",
    ]
    for folder in folders:
        folder_path = Path(folder)
        folder_suffix = folder_path.name.replace(common_prefix.name, "")
        if folder_suffix:
            folder_suffix = "_" + folder_suffix
        for tsv_file in folder_path.glob("linking_predictions*.tsv"):
            suffix = folder_suffix + tsv_file.stem.replace("linking_predictions", "")
            print(f"  Reading file {tsv_file} with suffix '{suffix}'")
            df = pd.read_csv(tsv_file, sep="\t")
            if "predicted_label" in df.columns:
                predicted_labels[suffix] = df["predicted_label"]
            # Remove empty columns
            df = df.dropna(axis=1, how="all")
            # Rename columns without names to avoid issues
            df = df.rename(columns=lambda x: "debugging_info" if x.startswith("Unnamed:") else x)
            system_columns = df.columns.difference(join_columns).tolist()
            # Add suffix to all system_columns
            df = df.rename(columns={c: c + suffix for c in system_columns})
            merged_data_frame = df if merged_data_frame is None else safe_merge(merged_data_frame, df, on=join_columns)
        for metrics_file in folder_path.glob("metrics*.json"):
            suffix = folder_suffix + metrics_file.stem.replace("metrics", "")
            print(f"  Reading file {metrics_file} with suffix '{suffix}'")
            df = pd.read_json(metrics_file, typ="series").to_frame().T.set_index(pd.Index([suffix]))
            merged_metrics = df if merged_metrics is None else pd.concat([merged_metrics, df])

    if merged_metrics is not None:
        merged_metrics_filename = "merged_metrics.tsv"
        merged_metrics.to_csv(merged_metrics_filename, sep="\t", index=True)
        print(f"Finished merging metrics files into {merged_metrics_filename}")

    if merged_data_frame is not None:
        gold_labels = merged_data_frame["label"]
        predicates = {}
        # Find boolean columns that can refine the accuracy calculation
        for col in merged_data_frame.columns:
            if not col.startswith("predicted_label") and merged_data_frame[col].dtype == bool:
                predicates[col] = merged_data_frame[col]
        accuracy_df = pd.DataFrame()
        for suffix, predicted_label in predicted_labels.items():
            accuracy = (predicted_label == gold_labels).mean()
            accuracy_df.loc[suffix, "accuracy"] = accuracy
            for pred_name, predicate in predicates.items():
                if predicate.any():
                    accuracy_pred = (predicted_label[predicate] == gold_labels[predicate]).mean()
                    accuracy_df.loc[suffix, f"accuracy_given_{pred_name}"] = accuracy_pred
        accuracy_filename = "merged_accuracies.tsv"
        accuracy_df.to_csv(accuracy_filename, sep="\t", index=True)
        print(f"Finished calculating accuracies into {accuracy_filename}")
        output_filename = "merged_linking_predictions.tsv"
        merged_data_frame.to_csv(output_filename, sep="\t", index=False)
        print(f"Finished merging files into {output_filename}")


if __name__ == "__main__":
    main()
