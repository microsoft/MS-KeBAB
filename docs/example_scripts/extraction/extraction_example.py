#!/usr/bin/env python3
"""Example extraction benchmark runner using an existing predictions file."""

import argparse
from pathlib import Path

from kebab import mskebab


def main():
    parser = argparse.ArgumentParser(description="Example extraction benchmark runner using an existing predictions file.")
    parser.add_argument("--predictions", type=Path, help="Path to predictions JSONL file", default=Path("data/predictions.jsonl"))
    parser.add_argument("--output_dir", type=Path, help="Optional output directory for results", default=Path("output_dir"))
    args = parser.parse_args()
    benchmark = mskebab.Benchmark(Path(__file__).parent / "tasks.json")
    task_instance = benchmark.tasks_by_name["Extraction-ReDocRED-Small"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_instance.evaluate(args.predictions, result_output_path=args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()