"""Bounded main-model retries, without changing auxiliary calls or tool execution."""

import pytest

from cs329z_hw1 import adapters
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.types import AgentConfig
from tests.fixture_tools import make_fixture_tools
from tests.helpers import truncated


def make_session(lm, **config):
    fx = make_fixture_tools()
    return adapters.run_agent_session(lm, fx.tools, AgentConfig(**config)), fx


def retries(result):
    return [e for e in result.transcript if e["type"] == "model_retry"]


def test_two_retries_can_recover_into_a_tool_call(scripted):
    lm = scripted(["", " \n\t", {"tool_call": {"name": "add", "args": {"a": 1, "b": 2}}}, "3"])
    session, fx = make_session(lm)
    result = session.send("Add 1 and 2.")
    assert result.status == "done" and result.text == "3"
    assert len(lm.calls) == 4 and lm.calls[0] == lm.calls[1] == lm.calls[2]
    assert fx.calls == [("add", {"a": 1, "b": 2})]
    assert [e["attempt"] for e in retries(result)] == [1, 2]
    assert [e["content"] for e in result.transcript if e["type"] == "assistant"][:2] == ["", " \n\t"]
    assert all(m["content"].strip() for m in session.messages)


def test_third_empty_response_fails_and_is_not_saved_as_a_fact(scripted, monkeypatch):
    lm = scripted(["", "", "", "Must not be requested"])
    session, fx = make_session(lm)
    saves = []
    monkeypatch.setattr(session, "_memory_save", lambda *args: saves.append(args))
    result = session.send("Find the spec.")
    assert result.status == "malformed" and "2 retries" in result.text
    assert len(lm.calls) == 3 and not fx.calls and not saves
    assert [e["status"] for e in retries(result)] == ["retrying", "retrying", "exhausted"]


@pytest.mark.parametrize("budget", ["turns", "tokens"])
def test_retry_respects_remaining_budget(scripted, budget):
    lm = scripted(["", "", "Must not be requested"])
    session, _ = make_session(lm)
    if budget == "turns":
        session.config.max_turns = 2
        expected_calls, expected_status = 2, "max_turns"
    else:
        session.config.token_budget = count_message_tokens([
            session.system_prompt, {"role": "user", "content": "Request"},
        ]) + count_tokens("")
        expected_calls, expected_status = 1, "token_budget"
    result = session.send("Request")
    assert result.status == expected_status and len(lm.calls) == expected_calls


def test_successful_tool_call_resets_empty_counter_without_reexecution(scripted):
    lm = scripted(["", "", {"tool_call": {"name": "add", "args": {"a": 1, "b": 2}}},
                   "", "", "3"])
    session, fx = make_session(lm)
    result = session.send("Add 1 and 2.")
    assert result.status == "done" and len(lm.calls) == 6
    assert fx.calls == [("add", {"a": 1, "b": 2})]
    assert lm.calls[3] == lm.calls[4] == lm.calls[5]
    assert [e["attempt"] for e in retries(result)] == [1, 2, 1, 2]


def test_new_send_gets_a_fresh_retry_counter(scripted):
    lm = scripted(["", "", "", "", "", "Recovered"])
    session, _ = make_session(lm)
    assert session.send("First request").status == "malformed"
    assert session.send("Second request").text == "Recovered"
    assert len(lm.calls) == 6


def test_empty_truncated_output_still_uses_existing_error_handling(scripted):
    lm = scripted([truncated(""), "Recovered"])
    session, _ = make_session(lm)
    result = session.send("Request")
    assert result.status == "done" and not retries(result)
    assert any(e["type"] == "error" and "truncated" in e["content"] for e in result.transcript)
