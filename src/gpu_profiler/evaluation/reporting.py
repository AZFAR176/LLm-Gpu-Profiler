from __future__ import annotations

import csv
from pathlib import Path

from gpu_profiler.evaluation.benchmarks import BenchmarkSummary


def write_benchmark_reports(summary: BenchmarkSummary, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary, out_dir / "benchmark_results.csv")
    _write_markdown(summary, out_dir / "benchmark_results.md")


def _write_csv(summary: BenchmarkSummary, file_path: Path) -> None:
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "expected", "predicted", "confidence", "top1_correct"])
        for r in summary.results:
            writer.writerow([r.scenario, r.expected, r.predicted, r.confidence, r.top1_correct])


def _write_markdown(summary: BenchmarkSummary, file_path: Path) -> None:
    lines = [
        "# Benchmark Results",
        "",
        f"- Total scenarios: {summary.total}",
        f"- Top-1 accuracy: {summary.top1_accuracy:.3f}",
        "",
        "| Scenario | Expected | Predicted | Confidence | Top-1 Correct |",
        "|---|---|---|---:|:---:|",
    ]
    for r in summary.results:
        lines.append(
            f"| {r.scenario} | {r.expected} | {r.predicted} | {r.confidence:.2f} | {'Y' if r.top1_correct else 'N'} |"
        )
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
