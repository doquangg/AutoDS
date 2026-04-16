from core.web.events import EventTypes, SeqCounter, build_event


def test_seq_counter_increments_monotonically():
    c = SeqCounter()
    assert c.next() == 1
    assert c.next() == 2
    assert c.next() == 3


def test_build_event_attaches_type_and_seq():
    seq = SeqCounter()
    ev = build_event(EventTypes.NODE_STARTED, seq, node="profiler", pass_count=0)
    assert ev["type"] == "node_started"
    assert ev["seq"] == 1
    assert ev["node"] == "profiler"
    assert ev["pass_count"] == 0


def test_build_event_includes_iso_timestamp():
    seq = SeqCounter()
    ev = build_event(EventTypes.HEARTBEAT, seq)
    assert "ts" in ev
    # ISO-8601 with timezone marker
    assert ev["ts"].endswith("Z") or "+" in ev["ts"]


def test_event_types_exposes_all_documented_kinds():
    expected = {
        "session_started", "node_started", "node_completed", "node_failed",
        "route_decision", "tool_call", "investigation_findings",
        "cleaning_recipe", "profile_summary", "model_metadata",
        "target_selection_required", "target_selection_resolved",
        "final_answer_token", "pipeline_complete", "pipeline_failed",
        "qa_token", "qa_complete", "heartbeat", "replay_truncated", "log",
    }
    actual = {getattr(EventTypes, name) for name in dir(EventTypes) if not name.startswith("_")}
    assert expected.issubset(actual)
