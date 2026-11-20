"""Boundary cases for request compaction, without network access."""

from copy import deepcopy

import pytest

from cs329z_hw1 import adapters
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.tokens import count_message_tokens
from cs329z_hw1.types import AgentConfig, ToolSpec


class Aux:
    def __init__(self, budget=800, response="User constraints retained."):
        self.budget = budget
        self.response = response
        self.calls = []

    def __call__(self, messages):
        assert count_message_tokens(messages) <= self.budget
        assert all(set(m) == {"role", "content"} for m in messages)
        self.calls.append(deepcopy(messages))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def session(lm, aux, budget=800, tools=None):
    result = adapters.run_agent_session(
        lm, tools or [], AgentConfig(context_budget=budget, aux_lm=aux),
    )
    result.system_prompt = {"role": "system", "content": "Assist the user."}
    return result


def test_exact_budget_does_not_compact(scripted):
    lm = scripted(["Done."])
    aux = Aux()
    agent = session(lm, aux)
    agent.config.context_budget = count_message_tokens([
        agent.system_prompt, {"role": "user", "content": "Request"},
    ])
    assert agent.send("Request").status == "done"
    assert aux.calls == []


def test_compaction_preserves_recent_pair_and_log_snapshot(scripted):
    lm = scripted(["Old response. " * 300, "Done."])
    aux = Aux()
    agent = session(lm, aux)
    first = agent.send("Budget 18450, Boulder not Oakland.")
    snapshot = deepcopy(agent.transcript)
    agent._append_message("assistant", "Recent call", "assistant")
    agent._append_message("user", "Recent result", "tool_result")
    assert agent.send("Current question").status == "done"
    assert aux.calls
    messages = lm.calls[-1]
    assert count_message_tokens(messages) <= 800
    contents = [m["content"] for m in messages]
    assert contents[-3:] == ["Recent call", "Recent result", "Current question"]
    assert agent.transcript[:len(snapshot)] == snapshot == first.transcript
    assert any("[user_request]" in call[1]["content"] for call in aux.calls)


@pytest.mark.parametrize("mode", ["normal", "oversized", "truncated"])
def test_big_feedback_is_recorded_exactly_as_shown(scripted, mode):
    response = {
        "normal": "Important result: 42.",
        "oversized": "Long summary. " * 1000,
        "truncated": OutputTruncated(
            "Partial summary: 42.", model="test", max_tokens=50, output_tokens=50,
        ),
    }[mode]
    aux = Aux(response=response)
    lm = scripted([{"tool_call": {"name": "big", "args": {}}}, "Done."])
    tools = [ToolSpec("big", "Large result", {"type": "object"}, lambda: "x" * 40_000)]
    agent = session(lm, aux, tools=tools)
    result = agent.send("Keep this exact user request.")
    assert result.status == "done"
    assert len(lm.calls) == 2
    assert aux.calls
    event = next(e for e in agent.transcript if e["type"] == "tool_result")
    assert event["content"] in [m["content"] for m in lm.calls[1]]
    assert len(event["content"]) < 40_000
    for messages in lm.calls:
        assert count_message_tokens(messages) <= 800
        assert any(m["content"] == "Keep this exact user request." for m in messages)


def test_empty_summary_stops_after_one_retry(scripted):
    aux = Aux(response="")
    lm = scripted(["Old response. " * 300, "Must not be called"])
    agent = session(lm, aux)
    agent.send("Old request")
    result = agent.send("Current request")
    assert result.status == "token_budget"
    assert "empty summary" in result.text
    assert len(aux.calls) == 2
    assert len(lm.calls) == 1


def test_aux_exception_propagates(scripted):
    agent = session(scripted(["Old response. " * 300]), Aux(response=RuntimeError("offline")))
    agent.send("Old request")
    with pytest.raises(RuntimeError, match="offline"):
        agent.send("Current request")


@pytest.mark.parametrize("oversized", ["system", "user"])
def test_mandatory_context_cannot_fit(scripted, oversized):
    lm = scripted(["Must not be called"])
    aux = Aux()
    agent = session(lm, aux)
    if oversized == "system":
        agent.system_prompt["content"] = "s" * 4000
    result = agent.send("u" * 4000 if oversized == "user" else "Request")
    assert result.status == "token_budget"
    assert not lm.calls and not aux.calls


def test_unconfigured_aux_uses_main_model(scripted):
    lm = scripted(["Old response. " * 300, "Done."])
    summaries = []

    def shared(messages):
        if messages[0]["content"].startswith("Compress conversation data"):
            summaries.append(deepcopy(messages))
            return "Remembered facts."
        return lm(messages)

    agent = session(shared, None)
    agent.send("Old request")
    assert agent.send("Current request").status == "done"
    assert summaries
    assert len(lm.calls) == 2
    assert len([e for e in agent.transcript if e["type"] == "assistant"]) == 2
