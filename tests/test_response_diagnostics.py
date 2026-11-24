"""Response diagnostics against real SDK objects and fake HTTP/SSE."""

import json

import pytest

from cs329z_hw1 import llm
from tests.test_llm_backends import (
    MESSAGES, api, bridge, chat_reply, isolated_env, terminal,
)


def records(tmp_path):
    return [json.loads(line) for line in (tmp_path / "raw_responses.jsonl").read_text(
        encoding="utf-8").splitlines()]


def delta(text):
    return {"type": "response.output_text.delta", "delta": text, "sequence_number": 0,
            "item_id": "msg_test", "output_index": 0, "content_index": 0}


def test_empty_terminal_preserves_streamed_text_for_diagnosis(api, bridge, tmp_path):
    api.replies.append([delta("Text that the terminal lost"), terminal("")])
    model = llm.LM(cache=False)
    assert model(MESSAGES) == ""  # The wrapper itself must not retry auxiliary calls.
    assert len(api.requests) == 1
    record, = records(tmp_path)
    assert record == model.last_response
    assert record["text"] == ""
    attempt, = record["attempts"]
    assert attempt["events"][0]["delta"] == "Text that the terminal lost"
    assert attempt["events"][-1]["response"]["status"] == "completed"
    assert attempt["response"]["output"][0]["content"][0]["text"] == ""
    assert attempt["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 3
    row, = llm.read_ledger()
    assert row["call_id"] == record["call_id"]


@pytest.mark.parametrize("native_tool", [False, True])
def test_errors_keep_raw_output_even_without_a_ledger_entry(api, bridge, tmp_path, native_tool):
    event = terminal()
    event["response"]["output"] = [{"type": "function_call", "id": "fc_test",
        "call_id": "call_test", "name": "search", "arguments": "{}", "status": "completed"}]
    api.replies.append([event] if native_tool else [delta("Partial")])
    model = llm.LM()
    with pytest.raises(RuntimeError, match="expects text only" if native_tool else "without a terminal"):
        model(MESSAGES)
    record, = records(tmp_path)
    assert record["error"].startswith("RuntimeError:")
    assert record["attempts"][0]["events"]
    if native_tool:
        assert record["attempts"][0]["response"]["output"][0]["type"] == "function_call"
    else:
        assert record["attempts"][0]["events"][0]["delta"] == "Partial"
    assert not llm.read_ledger()


def test_parameter_fallback_keeps_both_attempts(api, bridge, tmp_path):
    api.replies.extend([
        [{"type": "error", "sequence_number": 1, "code": "unsupported_parameter",
          "param": "temperature", "message": "Temperature is not supported"}],
        [terminal()],
    ])
    assert llm.LM()(MESSAGES) == "OK"
    record, = records(tmp_path)
    first, second = record["attempts"]
    assert first["events"][0]["type"] == "error" and first["error"]
    assert second["response"]["status"] == "completed"


def test_cache_hit_clears_previous_raw_record(api, bridge, tmp_path):
    api.replies.extend([[terminal("First")], [terminal("Second")]])
    model = llm.LM()
    assert model(MESSAGES) == "First"
    first_id = model.last_response["call_id"]
    assert model(MESSAGES) == "First"
    assert model.last_response is None
    assert len(records(tmp_path)) == 1
    assert model([{"role": "user", "content": "New input"}]) == "Second"
    assert model.last_response["call_id"] != first_id
    assert len(records(tmp_path)) == 2


def test_chat_completion_structure_is_recorded(api, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    api.replies.append(chat_reply())
    assert llm.LM(cache=False)(MESSAGES) == "OpenRouter reply"
    record, = records(tmp_path)
    assert record["attempts"][0]["response"]["choices"][0]["finish_reason"] == "stop"
    assert record["api_type"] == "chat_completions"
    assert "secret-test-key" not in json.dumps(record)


def test_truncation_preserves_terminal_status_and_details(api, bridge, tmp_path):
    api.replies.append([terminal("", status="incomplete", reason="max_output_tokens")])
    with pytest.raises(llm.OutputTruncated):
        llm.LM()(MESSAGES)
    record, = records(tmp_path)
    assert record["usage"]["finish_reason"] == "length"
    assert record["attempts"][0]["response"]["incomplete_details"]["reason"] == "max_output_tokens"


def test_diagnostic_write_failure_does_not_discard_the_answer(api, bridge, tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.open

    def fail_raw_log(path, *args, **kwargs):
        if path.name == "raw_responses.jsonl":
            raise OSError("Diagnostic disk unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_raw_log)
    api.replies.append([terminal()])
    model = llm.LM(cache=False)
    assert model(MESSAGES) == "OK"
    assert "Diagnostic disk unavailable" in model.last_response["logging_error"]
    assert len(llm.read_ledger()) == 1
