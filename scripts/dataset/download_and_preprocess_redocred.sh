#! /usr/bin/env bash

set -euo pipefail

# This script downloads and preprocesses the Redocred dataset.

# Usage: ./download_and_preprocess_redocred.sh [-f]

data_dir="./data"
date=$(date -u "+%Y-%m-%d")
wikidata_output_dir="${data_dir}/Wikidata/${date}"
redocred_output_dir="${data_dir}/Re-DocRED"
mkdir -p "$wikidata_output_dir"
wikidata_properties_file="wikidata_properties.json"
wikidata_properties_path="${wikidata_output_dir}/${wikidata_properties_file}"
regenerate_everything=false

if [[ "$#" -eq 1 ]] && [[ "$1" = "-f" ]]; then
    regenerate_everything=true
fi

if [ $regenerate_everything = "true" ] || [ ! -f $wikidata_properties_path ]; then
    echo "Fetching Wikidata properties..."
    uv run ./scripts/dataset/run_wikidata_simple_extractor.py --run-property-fetch --output-dir "$wikidata_output_dir"
else
    echo "Using existing Wikidata properties at ${wikidata_properties_path}"
fi

# Download the Redocred dataset
echo "Downloading Redocred dataset..."
for split in train dev test; do
    mkdir -p "${redocred_output_dir}/Original Dataset/${split}"
    split_filename="${redocred_output_dir}/Original Dataset/${split}/${split}.json"
    if [ $regenerate_everything = "true" ] || [ ! -f "$split_filename" ]; then
        curl -X GET "https://raw.githubusercontent.com/tonytan48/Re-DocRED/refs/heads/main/data/${split}_revised.json" \
        -o "$split_filename"
    else
        echo "File ${split_filename} already exists. Skipping download"
    fi
done
echo "Redocred dataset downloaded to ${redocred_output_dir}/Original Dataset"

# Preprocess the Redocred dataset
echo "Preprocessing Redocred dataset..."
for split in train dev test; do
    echo "Processing split: $split"
    extractions_path="${redocred_output_dir}/Extractions/${date}/${split}"
    if [ $regenerate_everything = "true" ] || [ ! -d $extractions_path ]; then
        rm -rf $extractions_path
        mkdir -p $extractions_path
    else
        echo "Extraction directory for ${split} already exists. Skipping preprocessing."
        continue
    fi
    uv run ./scripts/dataset/run_re_docred_dataset_builder.py \
    --re-docred-dir "${redocred_output_dir}/Original Dataset/${split}" \
    --wikidata-properties-path "$wikidata_properties_path" \
    --output-dir "${extractions_path}"
done

echo "Checking md5 of generated data..."
while IFS= read -r line || [[ -n "$line" ]]; do
    file=$(echo "$line" | awk '{print $2}')
    file_path=${redocred_output_dir}/Extractions/${date}/${file}
    expected_md5=$(echo "$line" | awk '{print $1}')
    actual_md5=$(tr -d "\r" < $file_path | md5sum | awk '{print $1}')

    if [ "$actual_md5" != "$expected_md5" ]; then
        echo "MD5 mismatch for ${file_path}: expected ${expected_md5}, got ${actual_md5}"
        exit 1
    fi
done < "${data_dir}/Re-DocRED/extraction.md5"

echo "MD5 check passed. All files are correctly generated."
