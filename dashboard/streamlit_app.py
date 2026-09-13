from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_source import load_latest_output


def _fmt_pct(value: float | int | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.1f}%"


def _fmt_ms(value: float | int | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value):,.1f} ms"


def _fmt_tok_s(value: float | int | None) -> str:
    if value is None:
        return "NA"
    val = float(value)
    if abs(val) >= 1000:
        return f"{val / 1000:.1f}k tok/s"
    return f"{val:.0f} tok/s"


def _fmt_int(value: float | int | None) -> str:
    if value is None:
        return "NA"
    return f"{int(float(value)):,}"


def _render_bar(values: dict, title: str, y_label: str) -> None:
    if not values:
        return
    df = pd.DataFrame(
        [{"name": str(k), "value": float(v)} for k, v in values.items() if v is not None]
    ).set_index("name")
    st.caption(title)
    st.bar_chart(df, height=220)
    st.caption(y_label)


def _top_banner(output: dict) -> None:
    state = output.get("state", {})
    diagnosis = output.get("diagnosis", {})
    context = state.get("context", {})
    run_id = state.get("run_id") or context.get("run_id") or "unknown-run"
    model_name = context.get("model_name", "unknown-model")
    mode = context.get("framework", "unknown-framework")
    bottleneck = diagnosis.get("primary_bottleneck", "NA")
    conf = diagnosis.get("confidence")

    c1, c2 = st.columns(2)
    c1.metric("Run ID", run_id)
    c2.metric("Model / Framework", f"{model_name} / {mode}")
    c3, c4 = st.columns(2)
    c3.metric("Current Bottleneck", bottleneck)
    c4.metric("Confidence", f"{float(conf):.2f}" if conf is not None else "NA")


def _telemetry_page(state: dict) -> None:
    st.subheader("Page 1: Live Telemetry")
    st.caption("Primary KPIs")
    a1, a2 = st.columns(2)
    a1.metric("Throughput", _fmt_tok_s(state.get("throughput_toks_per_sec")))
    a2.metric("Step Latency", _fmt_ms(state.get("step_latency_ms")))
    a3, a4 = st.columns(2)
    a3.metric("MFU", _fmt_pct(state.get("mfu_pct")))
    a4.metric("GPU Util", _fmt_pct(state.get("gpu_util_pct")))

    st.caption("System Signals")
    b1, b2 = st.columns(2)
    b1.metric("HBM BW", _fmt_pct(state.get("hbm_bw_pct")))
    b2.metric("NCCL Time", _fmt_pct(state.get("nccl_time_pct")))
    b3, b4 = st.columns(2)
    b3.metric("GPU Idle", _fmt_pct(state.get("gpu_idle_pct")))
    b4.metric("Sequence Length", _fmt_int(state.get("sequence_length")))

    by_rank_nccl = state.get("nccl_fraction_by_rank") or {}
    _render_bar(by_rank_nccl, "NCCL Fraction by Rank", "Per-rank share (%)")

    by_rank_idle = state.get("idle_fraction_by_rank") or {}
    _render_bar(by_rank_idle, "GPU Idle by Rank", "Per-rank share (%)")

    phase_breakdown = state.get("phase_breakdown_pct") or {}
    _render_bar(phase_breakdown, "Phase Breakdown (%)", "Phase contribution (%)")


def _diagnosis_page(diagnosis: dict, findings: list[dict], llm_calls: list[dict]) -> None:
    st.subheader("Page 2: Agent Diagnosis")
    st.write(f"**Primary bottleneck:** {diagnosis.get('primary_bottleneck', 'NA')}")
    st.write(f"**Confidence:** {diagnosis.get('confidence', 'NA')}")
    st.write(f"**Ask for more profiling:** {diagnosis.get('ask_for_more_profiling', False)}")

    st.write("**Evidence**")
    for item in diagnosis.get("evidence", []):
        st.write(f"- {item}")
    if diagnosis.get("evidence_ids"):
        st.write("**Evidence IDs**")
        for eid in diagnosis.get("evidence_ids", []):
            st.code(eid)

    st.write("**Missing evidence**")
    for item in diagnosis.get("missing_evidence", []):
        st.write(f"- {item}")

    st.write("**Recommended actions**")
    for item in diagnosis.get("recommended_actions", []):
        st.write(f"- {item}")

    exp = diagnosis.get("next_experiment")
    if exp:
        st.write("**Next experiment**")
        st.json(exp)

    if findings:
        st.write("**Detector Findings**")
        st.json(findings)

    if llm_calls:
        st.write("**LLM Call Logs**")
        st.json(llm_calls)


def _validation_page(validation: dict | None, output: dict) -> None:
    st.subheader("Page 3: Validation + Usage")
    if not validation:
        st.info("No before/after validation data yet.")
        st.markdown(
            "To populate this page, run with `--validate-input ./sample_data/after` and emit dashboard JSON."
        )
    else:
        c1, c2 = st.columns(2)
        c1.metric("Improved", str(validation.get("improved")))
        c2.metric("Summary", validation.get("summary", "NA"))
        st.write("**Deltas**")
        st.json(validation.get("deltas", {}))

    st.divider()
    st.write("### Quick Commands")
    st.code(
        "PYTHONPATH=src python -m gpu_profiler.main "
        "--input ./sample_data/before "
        "--telemetry-provider file "
        "--dashboard-output-dir ./dashboard_data",
        language="bash",
    )
    st.code("streamlit run dashboard/streamlit_app.py --server.port 8501", language="bash")
    st.write("### Current Output Snapshot")
    st.json(
        {
            "diagnosis_source": output.get("diagnosis_source"),
            "has_validation": output.get("validation") is not None,
            "executed_actions": len(output.get("executed_profiling_actions", [])),
        }
    )


def _a3_summary_page(output: dict) -> None:
    state = output.get("state", {})
    diagnosis = output.get("diagnosis", {})
    validation = output.get("validation")

    st.subheader("A3 Summary: One-Page View")
    st.caption("Designed for screenshot and print-style review")

    st.markdown("### 1) Current Performance")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Throughput", _fmt_tok_s(state.get("throughput_toks_per_sec")))
    p2.metric("Step Latency", _fmt_ms(state.get("step_latency_ms")))
    p3.metric("MFU", _fmt_pct(state.get("mfu_pct")))
    p4.metric("GPU Util", _fmt_pct(state.get("gpu_util_pct")))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("HBM BW", _fmt_pct(state.get("hbm_bw_pct")))
    s2.metric("NCCL Time", _fmt_pct(state.get("nccl_time_pct")))
    s3.metric("GPU Idle", _fmt_pct(state.get("gpu_idle_pct")))
    s4.metric("Seq Length", _fmt_int(state.get("sequence_length")))

    c_left, c_mid, c_right = st.columns(3)
    with c_left:
        _render_bar(state.get("nccl_fraction_by_rank") or {}, "NCCL by Rank", "Share (%)")
    with c_mid:
        _render_bar(state.get("idle_fraction_by_rank") or {}, "Idle by Rank", "Share (%)")
    with c_right:
        _render_bar(state.get("phase_breakdown_pct") or {}, "Phase Breakdown", "Share (%)")

    st.markdown("### 2) Diagnosis")
    d1, d2 = st.columns(2)
    d1.metric("Primary Bottleneck", diagnosis.get("primary_bottleneck", "NA"))
    d2.metric(
        "Confidence",
        f"{float(diagnosis.get('confidence')):.2f}" if diagnosis.get("confidence") is not None else "NA",
    )
    st.write("**Top Evidence**")
    evidence = diagnosis.get("evidence", [])[:5]
    if evidence:
        for item in evidence:
            st.write(f"- {item}")
    else:
        st.write("- NA")
    st.write("**Recommended Actions**")
    actions = diagnosis.get("recommended_actions", [])[:5]
    if actions:
        for item in actions:
            st.write(f"- {item}")
    else:
        st.write("- NA")

    st.markdown("### 3) Validation")
    if not validation:
        st.info("No before/after validation yet. Add `--validate-input ./sample_data/after`.")
    else:
        v1, v2 = st.columns(2)
        v1.metric("Improved", str(validation.get("improved")))
        v2.metric("Validation Summary", validation.get("summary", "NA"))
        deltas = validation.get("deltas", {})
        if deltas:
            st.write("**Top Deltas**")
            st.json(deltas)


def main() -> None:
    st.set_page_config(page_title="LLM GPU Profiler Dashboard", layout="wide")
    st.title("LLM GPU Profiler Dashboard")
    st.markdown(
        """
        <style>
        div[data-testid="metric-container"] {
            padding: 0.4rem 0.6rem;
        }
        div[data-testid="metric-container"] > label {
            font-size: 0.80rem;
        }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.20rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    default_path = str(Path.cwd() / "dashboard_data")
    data_dir = Path(st.sidebar.text_input("Dashboard data directory", value=default_path))
    refresh_s = st.sidebar.slider("Refresh interval (seconds)", min_value=1, max_value=10, value=2)
    page = st.sidebar.radio(
        "Select page",
        (
            "A3 Summary - One Page",
            "Page 1 - Live Telemetry",
            "Page 2 - Agent Diagnosis",
            "Page 3 - Validation + Usage",
        ),
    )
    st.sidebar.caption(
        "Emit data via CLI with --dashboard-output-dir to update latest_output.json."
    )

    output = load_latest_output(data_dir)
    if not output:
        st.warning("No dashboard snapshot found yet. Run pipeline with --dashboard-output-dir.")
    else:
        state = output.get("state", {})
        diagnosis = output.get("diagnosis", {})
        validation = output.get("validation")
        findings = output.get("findings", [])
        llm_calls = output.get("llm_calls", [])
        _top_banner(output)

        if page == "A3 Summary - One Page":
            _a3_summary_page(output)
        elif page == "Page 1 - Live Telemetry":
            _telemetry_page(state)
        elif page == "Page 2 - Agent Diagnosis":
            _diagnosis_page(diagnosis, findings, llm_calls)
        else:
            _validation_page(validation, output)

    st.caption(f"Auto-refresh every {refresh_s}s. Manual refresh: press R.")


if __name__ == "__main__":
    main()
