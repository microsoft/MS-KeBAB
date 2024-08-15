# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import pathlib
import random

from kebab.extraction.metrics import run_metrics_computation
from kebab.utils import save_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path to the folder with generated output json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Path to file where the metrics are saved.",
    )
    parser.add_argument("--logdir", type=pathlib.Path, required=True, help="Path to dir for the log files.")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Entity matching threshold.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logdir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Metrics Computation")
    logger.info(args)

    try:
        json_files = [x for x in args.input.glob("**/*.json") if x.is_file()]
        run_metrics_computation.run(args, logger, json_files)
        logger.info("Done evaluating generated entities.")

        random.seed(1234)
        sampled_json_files = random.sample(json_files, 20)
        run_metrics_computation.run(args, logger, sampled_json_files, "_sampled")
        data = []
        for file_name in sampled_json_files:
            with open(file_name) as file:
                data.append(json.load(file))
        save_json(json.dumps(data), args.output_dir / "sampled_files.json")
        logger.info("Done evaluating sampled generated entities.")

    except Exception as e:
        # Must explicitly log the exception for it to appear in logs in Heron.
        logger.exception(e)  # noqa: TRY401
        raise
