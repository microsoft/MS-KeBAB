import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kebab import mskebab


def assert_dicts_almost_equal(dict1, dict2, tol=1e-6):
    """Assert that two dictionaries with numeric values are almost equal."""
    assert dict1.keys() == dict2.keys(), "Dictionaries have different keys"
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]
        if isinstance(val1, dict) and isinstance(val2, dict):
            assert_dicts_almost_equal(val1, val2, tol)
        else:
            assert abs(val1 - val2) < tol, f"Values for key '{key}' differ: {val1} vs {val2}"

@dataclass
class FunctionDefinition:
    name: str
    arg_count: int
    callable: Callable

class FormulaCalculator:

    def __init__(self, supported_functions: dict[str, FunctionDefinition] | None = None):
        """Initialize the FormulaCalculator with supported functions."""
        self.supported_functions = {
            "+": FunctionDefinition(name="+", arg_count=2, callable=lambda x, y: x + y),
            "-": FunctionDefinition(name="-", arg_count=2, callable=lambda x, y: x - y),
            "*": FunctionDefinition(name="*", arg_count=2, callable=lambda x, y: x * y),
            "/": FunctionDefinition(name="/", arg_count=2, callable=lambda x, y: x / y if y != 0 else None),
            "SUM": FunctionDefinition(name="SUM", arg_count=-1, callable=sum),
            "AVERAGE": FunctionDefinition(name="AVERAGE", arg_count=-1, callable=lambda args: sum(args) / len(args) if args else 0),
        }
        if supported_functions:
            self.supported_functions.update(supported_functions)

    def tokenize(self, formula: str) -> list[str]:
        """Tokenize a formula string into components."""
        tokens = []
        current_token = ""
        for char in formula:
            if char in "+-*/(),":
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            elif char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
        if current_token:
            tokens.append(current_token)
        return tokens

    def is_float(self, token: str) -> bool:
        """Check if a token can be converted to a float."""
        try:
            float(token)
            return True
        except ValueError:
            return False

    def is_cell_or_range_reference(self, token: str) -> bool:
        """Check if a token is a cell or range reference (e.g., A1, B2:D4)."""
        # Simple check: starts with letters followed by numbers, possibly with a colon for ranges
        if ":" in token:
            parts = token.split(":")
            return all(part[0].isalpha() and part[1:].isdigit() for part in parts)
        return token[0].isalpha() and token[1:].isdigit()

    def lookup_cell_value(self, token: str, spreadsheet: any) -> float:
        """Lookup the value of a cell or range reference in the spreadsheet context."""
        # Placeholder implementation; actual implementation would depend on spreadsheet structure
        return float(spreadsheet.get(token, 0))

    def evaluate_operation(self, output_stack: list[str], operator_name: str, num_commas: int, spreadsheet: Any) -> None:
        arg_count = self.supported_functions[operator_name].arg_count
        if arg_count == -1:
            arg_count = num_commas + 1
            num_commas = 0 # reset for next function
        args = []
        for _ in range(arg_count):
            last_value = output.pop()
            args.append(float(last_value) if self.is_float(last_value) else self.lookup_cell_value(last_value, spreadsheet))
        args.reverse()
        result = self.supported_functions[operator_name].callable(*args)
    
    def evaluate_formula(self, tokens: list[str], spreadsheet: Any) -> list[str]:
        """Evaluate an excel-like formula given a list of tokens and a spreadsheet context.
            Use the shunting yard algorithm to convert infix to postfix notation.
            Limitations: doesn't handle nested functions."""
        precedence = defaultdict(lambda: 3)
        for key, value in {"+": 1, "-": 1, "*": 2, "/": 2}.items():
            precedence[key] = value
        output = []
        operators = []
        num_commas = 0
        for token in tokens:
            if token.isnumeric() or self.is_float(token):
                output.append(token)
            elif token in self.supported_functions:
                while (operators and operators[-1] != "(" and operators[-1] != "," and
                       (precedence[operators[-1]] >= precedence[token])):
                    operator_name = operators.pop()
                    arg_count = self.supported_functions[operator_name].arg_count
                    if arg_count == -1:
                        arg_count = num_commas + 1
                        num_commas = 0 # reset for next function
                    args = []
                    for _ in range(arg_count):
                        last_value = output.pop()
                        args.append(float(last_value) if self.is_float(last_value) else self.lookup_cell_value(last_value, spreadsheet))
                    args.reverse()
                    result = self.supported_functions[operator_name].callable(*args)
                    output.append(result)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()  # Remove '('
        while operators:
            output.append(operators.pop())
        return output
    


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
    compiler = ModelCompiler()
    model = compiler.read_and_parse_archive(debug_output_file, build_code=False)
    print(model.cells)
    evaluator = Evaluator(model)
    excel_metrics = {"property_precision": {}, "property_recall":{}}
    formula_cells = debug_output.formula_cells()
    for precision_cell, recall_cell in zip(formula_cells[0::2], formula_cells[1::2], strict=True):
        property_id_cell = precision_cell.address_at_offset(col_inc=-1)
        property_id = debug_output.evaluate(property_id_cell)
        precision = debug_output.evaluate(precision_cell)
        recall = debug_output.evaluate(recall_cell)
        if property_id != "Average":
            excel_metrics["property_precision"][property_id] = precision
            excel_metrics["property_recall"][property_id] = recall
        else:
            excel_metrics["avg_property_precision"] = precision # type: ignore
            excel_metrics["avg_property_recall"] = recall # type: ignore
    print(json.dumps(excel_metrics, indent=2))
    # Compare metrics
    assert_dicts_almost_equal(metrics["aesop"]["dataset_metrics"]["property_precision"], excel_metrics["property_precision"])
    assert_dicts_almost_equal(metrics["aesop"]["dataset_metrics"]["property_recall"], excel_metrics["property_recall"])
    assert abs(metrics["aesop"]["dataset_metrics"]["avg_property_precision"] - excel_metrics["avg_property_precision"]) < 1e-6
    assert abs(metrics["aesop"]["dataset_metrics"]["avg_property_recall"] - excel_metrics["avg_property_recall"]) < 1e-6


def run_evaluation(debug_data_path: Path):
    benchmark = mskebab.Benchmark(config_path=debug_data_path / "tasks.json",
                                  root_for_relative_paths=debug_data_path)
    task_instance = benchmark.tasks_by_name["Extraction-Debug-Info-Test"]
    predictions_file = debug_data_path / "predictions.jsonl"
    output_dir = debug_data_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_instance.evaluate(predictions_file, result_output_path=output_dir / "metrics.json")

