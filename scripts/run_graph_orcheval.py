"""
Run the full AutoDS pipeline with OrchEval tracing.

This script mirrors scripts/run_graph.py but layers OrchEval's LangGraph
adapter on top to capture structured trace events, generate reports, and
save everything for analysis.

What it does beyond run_graph.py:
  1. Hooks OrchEval's LangGraphAdapter into the pipeline via callbacks
  2. After pipeline completion, injects PassBoundary events from AutoDS's
     pass_history (since LangGraphAdapter can't auto-detect pass boundaries)
  3. Generates all five OrchEval reports (cost, timeline, routing, convergence, retries)
  4. Saves the raw trace as JSON for later analysis
  5. Prints a structured diagnostic summary

Configuration (environment variables):
  OPENAI_API_KEY       OpenAI API key (required)
  INVESTIGATOR_MODEL   Model for the investigator agent
  CODEGEN_MODEL        Model for the code generator agent
  EVALUATOR_MODEL      Model for the evaluator agent
  ANSWER_MODEL         Model for the answer agent
  AUTO_TARGET_COLUMN   Skip interactive target selection
  AUTODS_VERBOSE       Enable verbose logging (1 or full)

Usage:
  export OPENAI_API_KEY=sk-...
  export AUTO_TARGET_COLUMN=total_charge   # optional: skip interactive prompt
  python scripts/run_graph_orcheval.py

Requires:
  pip install orcheval  (or install from local source)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# AutoDS imports
# ---------------------------------------------------------------------------
from core.pipeline.graph import app  # noqa: E402
from core.logger import setup_logger  # noqa: E402

# ---------------------------------------------------------------------------
# OrchEval imports
# ---------------------------------------------------------------------------
try:
    from orcheval import Tracer, Trace, report
    from orcheval.events import PassBoundary
    from orcheval.report import (
        cost_report,
        routing_report,
        timeline_report,
        convergence_report,
        retry_report,
    )
except ImportError:
    sys.exit(
        "ERROR: orcheval is not installed.\n"
        "Install it with: pip install orcheval\n"
        "Or from local source: pip install -e /path/to/orcheval"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = (
    REPO_ROOT / "data" / "sample_data" / "healthcare"
    / "dirty_healthcare_visits_no_notes.csv"
)
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_CSV = OUTPUT_DIR / "cleaned_data.csv"
OUTPUT_VERBOSE_LOG = OUTPUT_DIR / "verbose.log"
OUTPUT_TRACE_JSON = OUTPUT_DIR / "orcheval_trace.json"
OUTPUT_REPORT_JSON = OUTPUT_DIR / "orcheval_report.json"

USER_QUERY = "What patterns in patient visits predict high-cost outcomes?"


# ---------------------------------------------------------------------------
# PassBoundary injection
# ---------------------------------------------------------------------------

def inject_pass_boundaries(
    trace: Trace,
    pass_history: list[dict[str, Any]],
    trace_id: str,
) -> Trace:
    """
    Create PassBoundary events from AutoDS's pass_history and merge them
    into the OrchEval trace.

    AutoDS tracks per-pass metadata in state["pass_history"], each entry:
        {
            "pass_number": int,
            "violations_found": int,
            "quality_score": float | None,       # algorithmic (deterministic)
            "steps_executed": int,
            "rows_after": int,
        }

    The LangGraphAdapter can't detect pass boundaries automatically (this is
    a documented OrchEval limitation). So we reconstruct them post-hoc from
    the pass_history and the trace's node-level timestamps.

    Strategy:
    - For each pass, find the first NodeEntry (profiler) and last NodeExit
      (evaluator/sandbox) that belong to that pass by walking the timeline.
    - Emit PassBoundary(direction="enter") at the profiler entry time and
      PassBoundary(direction="exit") at the evaluator exit time.
    - Attach the pass_history metrics as the exit snapshot.

    If we can't reliably map passes to timestamps (e.g., single pass or
    ambiguous boundaries), we fall back to evenly spaced synthetic timestamps.
    """
    if not pass_history:
        return trace

    events = list(trace.events)
    timeline = trace.get_timeline()

    if not timeline:
        return trace

    # Collect profiler NodeEntry timestamps and re_profile/sandbox NodeExit
    # timestamps to bracket each pass.
    from orcheval.events import NodeEntry, NodeExit

    profiler_entries = [
        e for e in timeline
        if isinstance(e, NodeEntry) and e.node_name == "profiler"
    ]
    re_profile_exits = [
        e for e in timeline
        if isinstance(e, NodeExit) and e.node_name == "re_profile"
    ]

    # Also grab sandbox exits as fallback (pass 0 might not have re_profile
    # if the pipeline fails early).
    sandbox_exits = [
        e for e in timeline
        if isinstance(e, NodeExit) and e.node_name == "sandbox"
    ]

    # Build time brackets from profiler entries so we can find the LAST
    # re_profile exit within each pass's window instead of using index-based
    # matching (which breaks when there are multiple exits per pass).
    bracket_ends = [pe.timestamp for pe in profiler_entries[1:]] + [timeline[-1].timestamp]

    new_events = []

    for i, ph in enumerate(pass_history):
        pass_num = ph.get("pass_number", i)

        # --- Enter timestamp: the profiler NodeEntry for this pass ---
        if i < len(profiler_entries):
            enter_ts = profiler_entries[i].timestamp
        else:
            # Fallback: offset from trace start
            enter_ts = timeline[0].timestamp

        # --- Exit timestamp: the LAST re_profile NodeExit within this pass's
        # time bracket (from this pass's profiler entry to the next one).
        bracket_end = bracket_ends[i] if i < len(bracket_ends) else timeline[-1].timestamp
        bracketed_re_profile_exits = [
            e for e in re_profile_exits
            if enter_ts <= e.timestamp <= bracket_end
        ]
        bracketed_sandbox_exits = [
            e for e in sandbox_exits
            if enter_ts <= e.timestamp <= bracket_end
        ]

        if bracketed_re_profile_exits:
            exit_ts = bracketed_re_profile_exits[-1].timestamp
        elif bracketed_sandbox_exits:
            exit_ts = bracketed_sandbox_exits[-1].timestamp
        else:
            exit_ts = bracket_end

        # --- Metrics snapshots ---
        # Enter snapshot: we don't have pre-pass metrics readily, so use
        # the previous pass's exit metrics (or empty for pass 0).
        if i > 0:
            prev = pass_history[i - 1]
            enter_metrics = {
                "violations_found": prev.get("violations_found", 0),
                "quality_score": prev.get("quality_score"),
                "rows_remaining": prev.get("rows_after"),
            }
        else:
            enter_metrics = {}

        exit_metrics = {
            "violations_found": ph.get("violations_found", 0),
            "quality_score": ph.get("quality_score"),
            "rows_remaining": ph.get("rows_after"),
            "steps_executed": ph.get("steps_executed", 0),
        }

        new_events.append(PassBoundary(
            trace_id=trace_id,
            timestamp=enter_ts,
            pass_number=pass_num,
            direction="enter",
            metrics_snapshot=enter_metrics,
            metadata={"source": "autods_pass_history", "injected": True},
        ))
        new_events.append(PassBoundary(
            trace_id=trace_id,
            timestamp=exit_ts,
            pass_number=pass_num,
            direction="exit",
            metrics_snapshot=exit_metrics,
            metadata={"source": "autods_pass_history", "injected": True},
        ))

    # Merge into existing events and return a new Trace
    all_events = events + new_events
    return Trace(events=all_events, trace_id=trace_id)


# ---------------------------------------------------------------------------
# Report display
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def display_cost_report(rpt) -> None:
    print_section("COST REPORT")
    total = rpt.total_cost
    tokens = rpt.total_tokens
    print(f"  Total cost          : {f'${total:.4f}' if total is not None else 'N/A (no cost data from provider)'}")
    print(f"  Total tokens        : {tokens.get('total', 0):,} "
          f"(prompt: {tokens.get('prompt', 0):,}, "
          f"completion: {tokens.get('completion', 0):,})")

    if rpt.most_expensive_node:
        print(f"  Most expensive node : {rpt.most_expensive_node}")
    if rpt.most_expensive_model:
        print(f"  Most expensive model: {rpt.most_expensive_model}")

    if rpt.nodes:
        print(f"\n  Per-node breakdown:")
        for node in rpt.nodes:
            cost_str = f"${node.total_cost:.4f}" if node.total_cost is not None else "N/A"
            print(f"    {node.node_name:<25s} {node.call_count} calls, "
                  f"{node.total_tokens:>7,} tokens, cost={cost_str}")
            for mu in node.models:
                mc = f"${mu.total_cost:.4f}" if mu.total_cost is not None else "N/A"
                dur = f"{mu.avg_duration_ms:.0f}ms" if mu.avg_duration_ms else "N/A"
                print(f"      {mu.model:<23s} {mu.call_count} calls, "
                      f"{mu.total_tokens:>6,} tok, cost={mc}, avg={dur}")


def display_timeline_report(rpt) -> None:
    print_section("TIMELINE REPORT")
    dur = rpt.total_duration_ms
    print(f"  Total duration      : {dur / 1000:.1f}s" if dur else "  Total duration      : N/A")
    print(f"  Node spans          : {len(rpt.spans)}")
    print(f"  Total events        : {len(rpt.events)}")

    if rpt.spans:
        print(f"\n  Span timeline:")
        for span in rpt.spans:
            d = f"{span.duration_ms:.0f}ms" if span.duration_ms else "?"
            children = len(span.children)
            print(f"    [{span.start_ms:>8.0f}ms] {span.node_name:<25s} "
                  f"duration={d}, children={children}")


def display_routing_report(rpt) -> None:
    print_section("ROUTING REPORT")
    print(f"  Total decisions     : {rpt.total_decisions}")
    print(f"  Unique sources      : {rpt.unique_sources}")
    print(f"  Unique targets      : {rpt.unique_targets}")

    if rpt.decisions:
        print(f"\n  Observed edges:")
        for edge in rpt.decisions:
            print(f"    {edge.source_node:<20s} → {edge.target_node:<20s} "
                  f"{edge.count}x ({edge.fraction:.0%})")

    if rpt.flags:
        print(f"\n  Flags ({len(rpt.flags)}):")
        for flag in rpt.flags:
            print(f"    [{flag.flag_type}] {flag.description}")
    else:
        print(f"\n  No suspicious routing patterns detected.")


def display_convergence_report(rpt) -> None:
    print_section("CONVERGENCE REPORT")
    print(f"  Total passes        : {rpt.total_passes}")
    conv = rpt.is_converging
    if conv is True:
        label = "YES — metrics improving across passes"
    elif conv is False:
        label = "NO — metrics not consistently improving"
    else:
        label = "INSUFFICIENT DATA (need ≥3 passes to determine)"
    print(f"  Is converging       : {label}")

    if getattr(rpt, "per_metric", None):
        print(f"\n  Per-metric convergence:")
        for mc in rpt.per_metric:
            print(f"    {mc.metric_name}: {mc.status}"
                  f"  (deltas: {', '.join(f'{d:.2f}' for d in mc.abs_deltas)})")

    if rpt.passes:
        print(f"\n  Pass summaries:")
        for p in rpt.passes:
            dur = f"{p.duration_ms:.0f}ms" if p.duration_ms else "N/A"
            print(f"    Pass {p.pass_number}: duration={dur}")
            if p.metrics_exit:
                for k, v in p.metrics_exit.items():
                    print(f"      {k}: {v}")
            if p.metric_deltas:
                deltas = ", ".join(f"{k}={v:+.2f}" for k, v in p.metric_deltas.items())
                print(f"      deltas: {deltas}")

    if rpt.metric_trends:
        print(f"\n  Metric trends (exit values across passes):")
        for metric, values in rpt.metric_trends.items():
            formatted = " → ".join(str(v) for v in values)
            print(f"    {metric}: {formatted}")

    if rpt.final_metrics:
        print(f"\n  Final metrics:")
        for k, v in rpt.final_metrics.items():
            print(f"    {k}: {v}")


def display_retry_report(rpt) -> None:
    print_section("RETRY REPORT")
    print(f"  Total errors        : {rpt.total_errors}")
    print(f"  Unique error types  : {rpt.unique_error_types}")
    rate = rpt.overall_retry_success_rate
    print(f"  Retry success rate  : {f'{rate:.0%}' if rate is not None else 'N/A (no retries detected)'}")

    if rpt.error_clusters:
        print(f"\n  Error clusters:")
        for cluster in rpt.error_clusters:
            print(f"    {cluster.error_type}: {cluster.count}x across nodes {cluster.nodes}")
            for msg in cluster.messages[:3]:
                print(f"      \"{msg[:100]}{'...' if len(msg) > 100 else ''}\"")

    if rpt.retry_sequences:
        print(f"\n  Retry sequences:")
        for seq in rpt.retry_sequences:
            status = "SUCCEEDED" if seq.succeeded else "FAILED"
            dur = f"{seq.total_retry_duration_ms:.0f}ms" if seq.total_retry_duration_ms else "?"
            print(f"    {seq.node_name}: {seq.attempt_count} attempts, "
                  f"{status}, duration={dur}")
    elif rpt.total_errors == 0:
        print(f"\n  No errors occurred — clean execution.")
    else:
        print(f"\n  Errors occurred but no retry sequences detected "
              f"(errors may have been in single-attempt nodes).")

    if rpt.nodes_with_errors:
        print(f"\n  Nodes with errors: {', '.join(rpt.nodes_with_errors)}")


def display_trace_summary(trace: Trace) -> None:
    print_section("TRACE SUMMARY")
    from orcheval.events import (
        NodeEntry, NodeExit, LLMCall, ToolCall,
        RoutingDecision, ErrorEvent, PassBoundary as PB,
    )

    event_counts: dict[str, int] = {}
    for e in trace:
        t = type(e).__name__
        event_counts[t] = event_counts.get(t, 0) + 1

    print(f"  Trace ID            : {trace.trace_id}")
    print(f"  Total events        : {len(trace)}")
    dur = trace.total_duration()
    print(f"  Wall-clock duration : {dur / 1000:.1f}s" if dur else "  Wall-clock duration : N/A")
    print(f"  Node sequence       : {' → '.join(trace.node_sequence())}")
    print(f"\n  Event type counts:")
    for etype, count in sorted(event_counts.items()):
        print(f"    {etype:<25s} {count}")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_report(full_report) -> dict[str, Any]:
    """Convert a FullReport to a JSON-serializable dict."""
    return {
        "cost": full_report.cost.model_dump(mode="json"),
        "routing": full_report.routing.model_dump(mode="json"),
        "convergence": full_report.convergence.model_dump(mode="json"),
        "timeline": {
            "total_duration_ms": full_report.timeline.total_duration_ms,
            "span_count": len(full_report.timeline.spans),
            "event_count": len(full_report.timeline.events),
            # Omit full spans/events to keep file manageable — the trace
            # JSON has all the raw data if needed.
        },
        "retries": full_report.retries.model_dump(mode="json"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=str(OUTPUT_VERBOSE_LOG))

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data from: {DATA_PATH}")
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows x {len(df.columns)} columns.\n")

    # ------------------------------------------------------------------
    # 2. Initialize OrchEval tracer
    # ------------------------------------------------------------------
    tracer = Tracer(adapter="langgraph", infer_routing=True, save_artifacts=True)
    print(f"OrchEval tracer initialized (trace_id={tracer.trace_id})")
    print(f"  Adapter: LangGraphAdapter (infer_routing=True)")
    print()

    # ------------------------------------------------------------------
    # 3. Run the AutoDS pipeline with OrchEval callbacks
    # ------------------------------------------------------------------
    initial_state = {
        "user_query": USER_QUERY,
        "working_df": df,
        "retry_count": 0,
        "tool_call_count": 0,
        "pass_count": 0,
        "target_column": None,
    }

    print("=" * 70)
    print("RUNNING AUTODS PIPELINE WITH ORCHEVAL TRACING")
    print("=" * 70)

    result = app.invoke(
        initial_state,
        config={"callbacks": [tracer.handler]},
    )

    # ------------------------------------------------------------------
    # 4. Collect trace and inject PassBoundary events
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ORCHEVAL ANALYSIS")
    print("=" * 70)

    raw_trace = tracer.collect()
    print(f"\nRaw trace collected: {len(raw_trace)} events")

    # Inject PassBoundary events from AutoDS's pass_history
    pass_history = result.get("pass_history", [])
    if pass_history:
        print(f"Injecting {len(pass_history)} pass boundaries from AutoDS state...")
        trace = inject_pass_boundaries(raw_trace, pass_history, tracer.trace_id)
        print(f"Enriched trace: {len(trace)} events "
              f"(+{len(trace) - len(raw_trace)} PassBoundary events)")
    else:
        trace = raw_trace
        print("No pass_history found — skipping PassBoundary injection.")

    # ------------------------------------------------------------------
    # 5. Display trace summary
    # ------------------------------------------------------------------
    display_trace_summary(trace)
    
    # ------------------------------------------------------------------
    # 5. Display trace in HTML form
    # ------------------------------------------------------------------
    trace.to_html("trace.html")

    # ------------------------------------------------------------------
    # 6. Generate and display all reports
    # ------------------------------------------------------------------
    full = report(trace)

    display_cost_report(full.cost)
    display_timeline_report(full.timeline)
    display_routing_report(full.routing)
    display_convergence_report(full.convergence)
    display_retry_report(full.retries)

    # ------------------------------------------------------------------
    # 7. Save artifacts
    # ------------------------------------------------------------------
    print_section("SAVED ARTIFACTS")

    # Save trace JSON
    trace_json = trace.to_json()
    OUTPUT_TRACE_JSON.write_text(trace_json)
    print(f"  Trace JSON          : {OUTPUT_TRACE_JSON}")
    print(f"                        ({len(trace)} events, "
          f"{len(trace_json) / 1024:.1f} KB)")

    # Save report JSON
    report_data = serialize_report(full)
    report_json = json.dumps(report_data, indent=2, default=str)
    OUTPUT_REPORT_JSON.write_text(report_json)
    print(f"  Report JSON         : {OUTPUT_REPORT_JSON}")

    # Save cleaned data (same as original run_graph.py)
    clean_df = result.get("clean_df")
    if clean_df is not None:
        clean_df.to_csv(OUTPUT_CSV, index=False)
        print(f"  Cleaned CSV         : {OUTPUT_CSV}")
        print(f"                        ({len(clean_df)} rows x "
              f"{len(clean_df.columns)} columns)")

    # ------------------------------------------------------------------
    # 8. AutoDS pipeline results (abbreviated)
    # ------------------------------------------------------------------
    print_section("AUTODS PIPELINE RESULTS")

    findings = result.get("investigation_findings")
    if findings:
        fd = findings.model_dump() if hasattr(findings, "model_dump") else findings
        print(f"  Target column       : {fd.get('target_column')}")
        print(f"  Task type           : {fd.get('task_type')}")
        print(f"  Violations found    : {len(fd.get('violations', []))}")
        print(f"  Columns to drop     : {fd.get('columns_to_drop', [])}")

    pass_count = result.get("pass_count", 0)
    profile = result.get("profile") or {}
    algo_score = (profile.get("algorithmic_quality_score") or {}).get("overall")
    score_str = f"{algo_score:.2f}" if algo_score is not None else "N/A"
    print(f"  Cleaning passes     : {pass_count}")
    print(f"  Final quality score : {score_str}")
    print(f"  Final answer        : {result.get('final_answer', 'N/A')}")

    if result.get("latest_error"):
        print(f"\n  LATEST ERROR: {result['latest_error']}")

    # ------------------------------------------------------------------
    # 9. Reload hint
    # ------------------------------------------------------------------
    print(f"\n{'─' * 70}")
    print(f"  To reload and re-analyze the trace later:")
    print(f"")
    print(f"    from orcheval import Trace, report")
    print(f"    trace = Trace.from_json(open('{OUTPUT_TRACE_JSON}').read())")
    print(f"    full = report(trace)")
    print(f"{'─' * 70}")
    print()


if __name__ == "__main__":
    main()