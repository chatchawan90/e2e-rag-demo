def test_monitor_records_metrics_without_keys_or_question_text(tmp_path):
    from envsearch.monitoring import record_event, recent_events
    record_event(tmp_path, "chat", "ok", 1.2, provider="anthropic", model="m",
                  question="private question", api_key="secret", usage={"input_tokens": 10, "output_tokens": 4},
                  trace={"returned": 2, "question": "private question"})
    events = recent_events(tmp_path)
    assert len(events) == 1 and events[0]["seconds"] == 1.2
    assert events[0]["usage"]["output_tokens"] == 4
    assert "secret" not in str(events) and "private question" not in str(events)


def test_monitor_history_persists_and_orders_latest_first(tmp_path):
    from envsearch.monitoring import record_event, recent_events
    record_event(tmp_path, "search", "ok", 0.1)
    record_event(tmp_path, "upload", "error", 0.2, error_type="ValueError")
    assert recent_events(tmp_path, limit=1)[0]["kind"] == "upload"
