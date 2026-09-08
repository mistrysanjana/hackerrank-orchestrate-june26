"""
Evaluation script for the Multi-Modal Evidence Review project.

Compares the generated output.csv against dataset/sample_claims.csv.

Usage from project root:
    python code/evaluation/main.py
"""

from __future__ import annotations

import csv
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

SAMPLE_CLAIMS_PATH = DATASET_DIR / "sample_claims.csv"
OUTPUT_PATH = PROJECT_ROOT / "output.csv"
REPORT_PATH = CODE_DIR / "evaluation_report.md"


OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


EVALUATED_COLUMNS = [
    "evidence_standard_met",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def normalize(value: str | None) -> str:
    if value is None:
        return ""

    return value.strip().lower()


def compare_values(
    expected: str,
    predicted: str,
) -> bool:
    return normalize(expected) == normalize(predicted)


def compare_risk_flags(
    expected: str,
    predicted: str,
) -> bool:
    expected_flags = {
        item.strip().lower()
        for item in expected.split(";")
        if item.strip()
    }

    predicted_flags = {
        item.strip().lower()
        for item in predicted.split(";")
        if item.strip()
    }

    return expected_flags == predicted_flags


def compare_image_ids(
    expected: str,
    predicted: str,
) -> bool:
    expected_ids = [
        item.strip()
        for item in expected.split(";")
        if item.strip()
    ]

    predicted_ids = [
        item.strip()
        for item in predicted.split(";")
        if item.strip()
    ]

    return expected_ids == predicted_ids


def evaluate(
    expected_rows: list[dict[str, str]],
    predicted_rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, int]], float]:
    predicted_by_user = {
        row.get("user_id", ""): row
        for row in predicted_rows
    }

    metrics: dict[str, dict[str, int]] = {}

    total_correct = 0
    total_comparisons = 0

    for column in EVALUATED_COLUMNS:
        metrics[column] = {
            "correct": 0,
            "total": 0,
        }

    for expected in expected_rows:
        user_id = expected.get("user_id", "")
        predicted = predicted_by_user.get(user_id)

        if predicted is None:
            continue

        for column in EVALUATED_COLUMNS:
            expected_value = expected.get(column, "")
            predicted_value = predicted.get(column, "")

            if column == "risk_flags":
                correct = compare_risk_flags(
                    expected_value,
                    predicted_value,
                )
            elif column == "supporting_image_ids":
                correct = compare_image_ids(
                    expected_value,
                    predicted_value,
                )
            else:
                correct = compare_values(
                    expected_value,
                    predicted_value,
                )

            metrics[column]["total"] += 1
            total_comparisons += 1

            if correct:
                metrics[column]["correct"] += 1
                total_correct += 1

    overall_accuracy = (
        total_correct / total_comparisons
        if total_comparisons
        else 0.0
    )

    return metrics, overall_accuracy


def build_report(
    expected_rows: list[dict[str, str]],
    predicted_rows: list[dict[str, str]],
    metrics: dict[str, dict[str, int]],
    overall_accuracy: float,
) -> str:
    lines: list[str] = []

    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(
        f"- Expected sample rows: {len(expected_rows)}"
    )
    lines.append(
        f"- Predicted rows: {len(predicted_rows)}"
    )
    lines.append(
        f"- Overall exact-field accuracy: "
        f"{overall_accuracy:.2%}"
    )
    lines.append("")

    lines.append("## Per-field Metrics")
    lines.append("")
    lines.append("| Field | Correct | Total | Accuracy |")
    lines.append("|---|---:|---:|---:|")

    for column in EVALUATED_COLUMNS:
        correct = metrics[column]["correct"]
        total = metrics[column]["total"]

        accuracy = correct / total if total else 0.0

        lines.append(
            f"| `{column}` | {correct} | {total} | "
            f"{accuracy:.2%} |"
        )

    lines.append("")
    lines.append("## Model Evaluation Strategy")
    lines.append("")
    lines.append(
        "The evaluation compares the generated predictions against "
        "`dataset/sample_claims.csv`."
    )
    lines.append("")
    lines.append(
        "Categorical fields are compared after trimming whitespace "
        "and normalizing case."
    )
    lines.append("")
    lines.append(
        "`risk_flags` are treated as unordered semicolon-separated "
        "sets."
    )
    lines.append("")
    lines.append(
        "`supporting_image_ids` are compared as ordered "
        "semicolon-separated identifiers."
    )
    lines.append("")

    lines.append("## Model Strategy Comparison")
    lines.append("")
    lines.append(
        "Strategy 1: single multimodal request containing the claim, "
        "history, evidence requirements, and all submitted images."
    )
    lines.append("")
    lines.append(
        "Strategy 2: multimodal request with stricter instructions "
        "to prioritize visual evidence, explicitly separate claim "
        "status from severity, and return only the required fields."
    )
    lines.append("")
    lines.append(
        "The production pipeline uses the stricter multimodal strategy "
        "because the task requires image-grounded decisions and "
        "structured output."
    )
    lines.append("")

    lines.append("## Cost and Runtime Measurement")
    lines.append("")
    lines.append(
        "The final report should record the approximate number of "
        "model calls, images processed, input/output tokens, runtime, "
        "and estimated cost after the sample and test runs."
    )
    lines.append("")
    lines.append(
        "Pricing assumptions should be recorded using the model pricing "
        "available at the time of execution."
    )
    lines.append("")

    lines.append("## Rate Limits and Reliability")
    lines.append("")
    lines.append(
        "The production pipeline should process claims sequentially "
        "or with controlled concurrency, use retry handling for "
        "transient API failures, and avoid unnecessary repeated "
        "requests."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print("Starting evaluation...")

    expected_rows = read_csv(SAMPLE_CLAIMS_PATH)

    if not OUTPUT_PATH.exists():
        print("ERROR: output.csv does not exist.")
        print(
            "Run the main pipeline first:"
        )
        print(
            "  python code\\main.py"
        )
        return

    predicted_rows = read_csv(OUTPUT_PATH)

    missing_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in predicted_rows[0]
    ] if predicted_rows else OUTPUT_COLUMNS

    if missing_columns:
        print("ERROR: output.csv is missing columns:")
        for column in missing_columns:
            print(f"  - {column}")
        return

    metrics, overall_accuracy = evaluate(
        expected_rows,
        predicted_rows,
    )

    report = build_report(
        expected_rows,
        predicted_rows,
        metrics,
        overall_accuracy,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Sample rows: {len(expected_rows)}")
    print(f"Predicted rows: {len(predicted_rows)}")
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    print()
    print(f"Report written to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
