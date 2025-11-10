# A command-line utility that will read all tsv files
# whose names start with "linking_predictions" in a folder
# and merge them into a single predictions file.
import sys
from pathlib import Path

import pandas as pd


def main():
    """Entry point for the script to generate merged linking predictions TSV files."""
    folders = sys.argv[1:]
    print(f"Merging linking predictions from folders: {folders}")
    output_filename = "merged_linking_predictions.tsv"
    merged_data_frame = None
    system_columns = ["log_odds", "predicted_label", "debugging_info"]
    for folder in folders:
        folder_path = Path(folder)
        tsv_files = list(folder_path.glob("linking_predictions*.tsv"))
        print(f"Merging {len(tsv_files)} files in folder {folder} into {output_filename}")
        for tsv_file in tsv_files:
            suffix = tsv_file.stem.replace("linking_predictions", "")
            print(f"  Reading file {tsv_file} with suffix '{suffix}'")
            df = pd.read_csv(tsv_file, sep="\t")
            # Remove empty columns
            df = df.dropna(axis=1, how="all")
            join_columns = df.columns.difference(system_columns).tolist()
            # Add suffix to all system_columns
            df = df.rename(columns={c: c + suffix for c in system_columns})
            merged_data_frame = df if merged_data_frame is None else merged_data_frame.merge(df, on=join_columns)
    if merged_data_frame is not None:
        merged_data_frame.to_csv(output_filename, sep="\t", index=False)
        print(f"Finished merging files into {output_filename}")

if __name__ == "__main__":
    main()
