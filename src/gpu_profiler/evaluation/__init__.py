"""Benchmark scenarios and evaluation utilities."""

from gpu_profiler.evaluation.benchmarks import run_benchmark_suite
from gpu_profiler.evaluation.reporting import write_benchmark_reports

__all__ = ["run_benchmark_suite", "write_benchmark_reports"]
