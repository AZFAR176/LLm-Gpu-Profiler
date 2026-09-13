# Roadmap

## Implemented Foundation (Current)

- common telemetry schema with snapshot + event support
- provider abstraction with `file` and `nvml` adapters
- time/rank-aware state builder with distributions and phase breakdown
- deterministic detector layer + evidence references
- LLM provider abstraction with retries + fallback
- policy-gated typed action execution
- synthetic benchmark harness for baseline diagnosis

## Expand Real Telemetry Adapters

- add DCGM adapter
- add PyTorch profiler adapter
- add NCCL trace adapter
- add Nsight and vLLM adapters

Deliverable: production telemetry coverage across training and inference stacks.

## Phase 2: LLM Runtime Hardening

- add provider-level fallback chain
- add prompt/version registry
- add richer usage and cost accounting
- add latency/error SLO metrics

Deliverable: robust LLM reasoning runtime with predictable failure handling.

## Phase 3: Detector Expansion + Evidence Graph

- add OOM risk, kernel fragmentation, KV pressure detectors
- unify detector evidence into graph-indexed references
- calibrate confidence using detector agreement and completeness

Deliverable: grounded multi-signal findings with calibrated confidence.

## Phase 4: Safe Experiment Layer

- add full typed experiment set (batch, TP, workers, FlashAttention, FSDP policy)
- add dry-run mode, rollback plans, and explicit approval gates
- track per-experiment runtime and resource cost

Deliverable: safe optimizer loop with policy controls.

## Phase 5: Evaluation + Validation at Scale

- express recommendations as experiments:
  - hypothesis
  - action/config change
  - measure set
  - stop condition
- add benchmark scenarios for known failure modes
- compute top-1/top-3 accuracy, false-positive rate, recommendation success rate
- compare baseline-vs-LLM diagnosis quality and cost

Deliverable: repeatable evidence for production readiness.

## Phase 6: Persistence + Deployment

- add run/diagnosis/experiment persistence
- add OpenTelemetry tracing and Prometheus metrics
- deploy as node agents + central reasoning service

Deliverable: production-grade operations and historical learning.
