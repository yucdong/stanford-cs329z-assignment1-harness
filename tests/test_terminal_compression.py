"""Terminal observation budgets and auxiliary summaries, without model/network calls."""

import pytest

from cs329z_hw1 import sandbox
from cs329z_hw1.cardinal import make_terminal_tool
from cs329z_hw1.sandbox import TerminalResult
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.types import AgentConfig, ToolRegistry


def execute(monkeypatch, result, **kwargs):
    monkeypatch.setattr(sandbox, "run_terminal", lambda *a, **k: result)
    registry = ToolRegistry([make_terminal_tool(AgentConfig(**kwargs))])
    return registry.execute("run_terminal", {"cmd": "echo example"})


@pytest.mark.parametrize("state, status", [
    ({"exit_code": 0}, "ok"),
    ({"exit_code": 2}, "tool_error"),
    ({"timed_out": True}, "tool_error"),
    ({"rejected": True}, "tool_error"),
])
def test_short_result_keeps_all_fields(monkeypatch, state, status):
    def unused(messages):
        pytest.fail("Short results need no summary")

    result = execute(monkeypatch, TerminalResult(
        stdout="partial output", stderr="diagnostic", reason="sandbox reason", **state,
    ), aux_lm=unused, context_budget=200)
    assert result.status == status
    for text in ("partial output", "diagnostic", "sandbox reason", "exit code:"):
        assert text in result.content
    if state.get("timed_out"):
        assert "timed out" in result.content


def test_large_result_summarizes_all_fragments_with_bounded_requests(monkeypatch):
    calls = []

    def aux(messages):
        assert count_message_tokens(messages) <= 6000
        calls.append(messages)
        return "Count: 123; date: 2001-06-27."

    raw = "x" * 600_000
    result = execute(monkeypatch, TerminalResult(
        exit_code=0, stdout=raw, stderr="warning", reason="detail",
    ), aux_lm=aux, context_budget=6000)
    assert calls
    assert "".join(call[1]["content"] for call in calls) == raw
    assert result.status == "ok"
    assert count_tokens(result.content) <= 6000
    for text in ("123", "2001-06-27", "warning", "detail", "exit code: 0"):
        assert text in result.content


@pytest.mark.parametrize("mode", ["missing", "raises", "oversized", "empty"])
def test_summary_fallback_preserves_status_and_budget(monkeypatch, mode):
    def aux(messages):
        if mode == "raises":
            raise RuntimeError("summarizer unavailable")
        return "" if mode == "empty" else "z" * 50_000

    result = execute(monkeypatch, TerminalResult(
        exit_code=3, stdout="o" * 20_000, stderr="e" * 20_000, reason="r" * 20_000,
    ), aux_lm=None if mode == "missing" else aux, context_budget=1000)
    assert result.status == "tool_error"
    assert count_tokens(result.content) <= 1000
    for label in ("exit code: 3", "stdout:", "stderr:", "reason:"):
        assert label in result.content


@pytest.mark.parametrize("budget", [0, 1, 10])
def test_tiny_budget_is_still_enforced(monkeypatch, budget):
    result = execute(monkeypatch, TerminalResult(exit_code=0, stdout="x" * 100),
                     context_budget=budget)
    assert result.status == "ok"
    assert count_tokens(result.content) <= budget
