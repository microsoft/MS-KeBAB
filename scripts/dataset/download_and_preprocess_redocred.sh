#! /usr/bin/env bash

set -euxo pipefail

# This script downloads and preprocesses the Redocred dataset.

# Usage: ./preprocess_redocred.sh [--fetch-wikidata-properties]

data_dir="${PWD}/data/"
wikidata_output_dir="${data_dir}/Wikidata"
redocred_output_dir="${data_dir}/Re-DocRED"
mkdir -p "$wikidata_output_dir"
wikidata_properties_file="wikidata_properties.json"
wikidata_properties_path="${wikidata_output_dir}/${wikidata_properties_file}"
fetch_properties=false

if [[ "$#" -eq 1 ]] && [[ "$1" = "--fetch-wikidata-properties" ]]; then
    fetch_properties=true
fi

if [ $fetch_properties = "true" ] || [ ! -f $wikidata_properties_path ]; then
    echo "Fetching Wikidata properties..."
    uv run ./scripts/dataset/run_wikidata_simple_extractor.py --run-property-fetch --output-dir "$wikidata_output_dir"
else
    echo "Using existing Wikidata properties at ${wikidata_properties_path}"
fi

# Download the Redocred dataset
echo "Downloading Redocred dataset..."
for split in train dev test; do
    mkdir -p "${redocred_output_dir}/${split}"
    split_filename="${redocred_output_dir}/${split}/${split}.json"
    if [ ! -f $split_filename ]; then
        curl -X GET "https://raw.githubusercontent.com/tonytan48/Re-DocRED/refs/heads/main/data/${split}_revised.json" \
        -o $split_filename
    else
        echo "File ${split_filename} already exists. Skipping download"
    fi
done
echo "Redocred dataset downloaded to ${redocred_output_dir}"

# Preprocess the Redocred dataset
echo "Preprocessing Redocred dataset..."
for split in train dev test; do
    echo "Processing split: $split"
    uv run ./scripts/dataset/run_re_docred_dataset_builder.py \
    --re-docred-dir "${redocred_output_dir}/${split}" \
    --wikidata-properties-path "$wikidata_properties_path" \
    --output-dir "${redocred_output_dir}/Extraction/${split}"
done

echo "Checking md5 of generated data..."
md5sum -c ./data/Re-DocRED/extraction.md5 --status