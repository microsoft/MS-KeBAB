#! /usr/bin/env bash

set -euo pipefail

# This script downloads and preprocesses the Redocred dataset.

# Usage: ./download_and_preprocess_redocred.sh [-f]

data_dir="./data"
wikidata_output_dir="${data_dir}/Wikidata"
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
    mkdir -p "${redocred_output_dir}/${split}"
    split_filename="${redocred_output_dir}/${split}/${split}.json"
    if [ $regenerate_everything = "true" ] || [ ! -f $split_filename ]; then
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
    if [ $regenerate_everything = "true" ] || [ ! -d "${redocred_output_dir}/Extraction/${split}" ]; then
        rm -rf "${redocred_output_dir}/Extraction/${split}"
        mkdir -p "${redocred_output_dir}/Extraction/${split}"
    else
        echo "Extraction directory for ${split} already exists. Skipping preprocessing."
        continue
    fi
    uv run ./scripts/dataset/run_re_docred_dataset_builder.py \
    --re-docred-dir "${redocred_output_dir}/${split}" \
    --wikidata-properties-path "$wikidata_properties_path" \
    --output-dir "${redocred_output_dir}/Extraction/${split}"
done

echo "Checking md5 of generated data..."
if md5sum -c ${data_dir}/Re-DocRED/extraction.md5 --status; then
    echo "MD5 check passed. All files are correctly generated."
else
    echo "MD5 check failed. Please check the generated files."
    exit 1
fi