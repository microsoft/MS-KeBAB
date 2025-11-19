import json
from pathlib import Path
from pycel import ExcelCompiler

from kebab import mskebab

def test_debug_output_consistency(debug_data_path: Path):
    """Test that debug output is consistent with evaluation metrics."""
    # Run evaluation to produce debug output
    run_evaluation(debug_data_path)

    # Load evaluation metrics
    metrics_file = debug_data_path / "output" / "metrics.json"
    with metrics_file.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Load debug output
    debug_output_file = debug_data_path / "output" / "debug_output" / "debug_info.xlsx"
    debug_output = ExcelCompiler(str(debug_output_file))
    excel_metrics = {}

    for cell in debug_output.formula_cells():
        print(cell)


def run_evaluation(debug_data_path: Path):
    benchmark = mskebab.Benchmark(config_path=debug_data_path / "tasks.json",
                                  root_for_relative_paths=debug_data_path)
    task_instance = benchmark.tasks_by_name["Extraction-Debug-Info-Test"]
    predictions_file = debug_data_path / "predictions.jsonl"
    output_dir = debug_data_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_instance.evaluate(predictions_file, result_output_path=output_dir / "metrics.json")

