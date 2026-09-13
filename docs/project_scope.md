# Project Scope

## Objective

Build an agentic performance system:

```text
Observe telemetry -> reason about bottlenecks -> propose experiments -> validate outcomes
```

The project targets both **training** and **inference** workloads.

## Design Principles

- Separate **telemetry collection** from **reasoning**.
- Normalize raw logs into a **structured performance state**.
- Ensure the LLM **reasons over measured facts** and never fabricates telemetry.
- Require evidence-backed, structured diagnostic output.
- Treat optimization advice as **hypothesis-driven experiments**.
- Validate improvements with before/after telemetry, not intuition.

## Inputs

- GPU utilization, SM/Tensor Core activity, MFU
- HBM bandwidth and memory usage
- NCCL communication time and collective overhead
- CPU/data-loader overhead
- GPU idle fraction
- Step latency, tokens/sec
- Batch size, sequence length, world size
- Kernel timing overhead

## Output Contract

1. Primary bottleneck and confidence
2. Evidence with exact supporting telemetry signals
3. Likely root causes
4. Missing evidence (what else must be profiled)
5. Recommended actions
6. Next experiment (hypothesis, action, measures, stop condition)
7. Optional validation report after re-run

## Missing-Evidence Behavior

When data is insufficient, the agent must return an explicit uncertainty statement and a profiling request.

Example:

```text
I cannot distinguish communication-bound vs underfed with current evidence.
Collect collective-level NCCL breakdown and GPU idle timeline for 100 steady-state steps.
```

## Bottleneck Classes

- compute bound
- memory bandwidth bound
- communication bound
- CPU/data pipeline bound
- synchronization bound
- GPU underfeeding / poor batching
- load imbalance / straggler
- mixed/unknown (requires more evidence)

## Non-goals (v0)

- Fully autonomous kernel-level tuning
- Guaranteed single-cause diagnosis for every run
- Full coverage for all telemetry backends on day one

## Success Metrics

- Diagnosis coverage across collected runs
- Agreement with expert labels on curated traces
- Fraction of experiments that improve latency/throughput/MFU
- Time-to-actionable recommendation
