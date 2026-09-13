from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_profiler.config import RunConfig
from gpu_profiler.evaluation.benchmarks import run_benchmark_suite
from gpu_profiler.pipeline import ProfilerPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-driven GPU profiling agent CLI")
    parser.add_argument("--input", type=Path, required=True, help="Baseline telemetry input directory")
    parser.add_argument(
        "--telemetry-provider",
        type=str,
        default="file",
        choices=["file", "nvml", "dcgm", "pytorch_profiler"],
        help="Telemetry source provider",
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    parser.add_argument(
        "--followup-input",
        type=Path,
        default=None,
        help="Optional additional profiling input when agent requests more evidence",
    )
    parser.add_argument(
        "--validate-input",
        type=Path,
        default=None,
        help="Optional post-change telemetry input for before/after validation",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=2,
        help="Maximum observe->reason rounds",
    )
    parser.add_argument(
        "--llm-response-file",
        type=Path,
        default=None,
        help="Optional raw LLM JSON response file to validate and use as diagnosis",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="disabled",
        choices=["disabled", "mock", "openai", "local_openai"],
        help="Live provider selection (ignored when --llm-response-file is used)",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="Model identifier")
    parser.add_argument("--llm-api-base", type=str, default=None, help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--llm-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable containing API key",
    )
    parser.add_argument("--llm-timeout-s", type=float, default=20.0)
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument("--llm-retry-backoff-s", type=float, default=1.5)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-output-tokens", type=int, default=1200)
    parser.add_argument("--llm-input-cost-per-mtok-usd", type=float, default=None)
    parser.add_argument("--llm-output-cost-per-mtok-usd", type=float, default=None)
    parser.add_argument(
        "--dashboard-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for live dashboard snapshot JSON files",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the strict prompt template for the current input state and exit",
    )
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run synthetic evaluation scenarios and print summary JSON",
    )
    parser.add_argument(
        "--benchmark-report-dir",
        type=Path,
        default=None,
        help="Optional output directory for benchmark CSV/Markdown reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        input_path=args.input,
        telemetry_provider=args.telemetry_provider,
        mode=args.mode,
        followup_path=args.followup_input,
        validate_path=args.validate_input,
        max_reasoning_rounds=args.max_rounds,
        llm_response_path=args.llm_response_file,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_api_base=args.llm_api_base,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout_s=args.llm_timeout_s,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff_s=args.llm_retry_backoff_s,
        llm_temperature=args.llm_temperature,
        llm_max_output_tokens=args.llm_max_output_tokens,
        llm_input_cost_per_mtok_usd=args.llm_input_cost_per_mtok_usd,
        llm_output_cost_per_mtok_usd=args.llm_output_cost_per_mtok_usd,
        dashboard_output_dir=args.dashboard_output_dir,
    )
    pipeline = ProfilerPipeline()
    if args.run_benchmarks:
        summary = run_benchmark_suite()
        print(json.dumps(summary.model_dump(mode="json"), indent=2))
        if args.benchmark_report_dir is not None:
            from gpu_profiler.evaluation.reporting import write_benchmark_reports

            write_benchmark_reports(summary, args.benchmark_report_dir)
        return

    if args.print_prompt:
        print(pipeline.build_prompt(cfg))
        return

    output = pipeline.run(cfg)
    print(json.dumps(output.model_dump(), indent=2))


if __name__ == "__main__":
    main()
