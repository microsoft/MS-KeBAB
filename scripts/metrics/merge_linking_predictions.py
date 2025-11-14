# A command-line utility that will read all tsv files
# whose names start with "linking_predictions" in a folder
# and merge them into a single predictions file.
import argparse
import os
from pathlib import Path

import pandas as pd


def main():
    """Entry point for the script to generate merged linking predictions TSV files."""
    parser = argparse.ArgumentParser(
        description="""
    This will merge all linking_predictions*.tsv files from the specified folders.
    The merged file will be named 'merged_linking_predictions.tsv'.
    Each system's columns will be suffixed with the folder name, excluding the common prefix.
    If a tsv file has a suffix in its name (after 'linking_predictions'), that suffix will also be added.
    """
    )
    parser.add_argument("folder", nargs="+", type=Path, help="folder to merge")
    args = parser.parse_args()
    folders = args.folder
    # Find the common prefix of all folder names
    common_prefix = Path(os.path.commonprefix(folders))
    output_filename = "merged_linking_predictions.tsv"
    merged_data_frame = None
    system_columns = ["log_odds", "predicted_label", "debugging_info"]
    for folder in folders:
        folder_path = Path(folder)
        folder_suffix = folder_path.name.replace(common_prefix.name, "")
        if folder_suffix:
            folder_suffix = "_" + folder_suffix
        for tsv_file in folder_path.glob("linking_predictions*.tsv"):
            suffix = folder_suffix + tsv_file.stem.replace("linking_predictions", "")
            print(f"  Reading file {tsv_file} with suffix '{suffix}'")
            df = pd.read_csv(tsv_file, sep="\t")
            # Remove empty columns
            df = df.dropna(axis=1, how="all")
            # We hope that join_columns are the same across dataframes and that a combined value of these columns uniquely identifies the row.
            join_columns = df.columns.difference(system_columns).tolist()
            # Add suffix to all system_columns
            df = df.rename(columns={c: c + suffix for c in system_columns})
            merged_data_frame = df if merged_data_frame is None else merged_data_frame.merge(df, on=join_columns)
    if merged_data_frame is not None:
        merged_data_frame.to_csv(output_filename, sep="\t", index=False)
        print(f"Finished merging files into {output_filename}")


if __name__ == "__main__":
    main()
