"""Local backend adaptation tests: real SDK, fake HTTP/SSE, no model calls."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest
from openai import OpenAI

from cs329z_hw1 import llm


MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
    {"role": "user", "content": "Reply OK."},
]


def terminal(text="OK", *, status="completed", reason=None, usage=True, model="returned-model"):
    response = {
        "id": "resp_test", "object": "response", "created_at": 1,
        "status": status, "model": model,
        "output": [{
            "id": "msg_test", "type": "message", "role": "assistant", "status": status,
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "usage": {
            "input_tokens": 30, "output_tokens": 10, "total_tokens": 40,
            "input_tokens_details": {"cached_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 3},
        } if usage else None,
        "incomplete_details": {"reason": reason} if reason else None,
    }
    return {"type": f"response.{status}", "sequence_number": 1, "response": response}


def chat_reply(*, cost=None):
    usage = {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
    if cost is not None:
        usage["cost"] = cost
    return httpx.Response(200, json={
        "id": "chat_test", "object": "chat.completion", "created": 1,
        "model": "openai/gpt-6-luna", "usage": usage,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "OpenRouter reply",
        }}],
    })


def rejection(parameter, *, status=400):
    return httpx.Response(status, json={"error": {
        "message": f"Unsupported parameter: '{parameter}' is not supported with this model.",
        "type": "invalid_request_error", "code": "unsupported_parameter", "param": parameter,
    }})


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_load_env", lambda: None)
    for key in list(os.environ):
        if key.startswith("CS329Z_") or key == "OPENROUTER_API_KEY":
            monkeypatch.delenv(key)
    monkeypatch.setenv("CS329Z_CACHE_DIR", str(tmp_path))
    llm._NO_TEMPERATURE.clear()
    llm._NO_EFFORT.clear()
    yield
    llm._NO_TEMPERATURE.clear()
    llm._NO_EFFORT.clear()


@pytest.fixture(autouse=True)
def api(monkeypatch):
    state = SimpleNamespace(replies=[], requests=[], responses=[], clients=[])

    def handle(request):
        state.requests.append(request)
        assert state.replies, "Unexpected HTTP request; tests must never use the network"
        reply = state.replies.pop(0)
        if isinstance(reply, list):
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in reply)
            reply = httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        state.responses.append(reply)
        return reply

    def client(**kwargs):
        kwargs["max_retries"] = 0
        kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handle))
        instance = OpenAI(**kwargs)
        state.clients.append(instance)
        return instance

    monkeypatch.setattr(openai, "OpenAI", client)
    yield state
    for instance in state.clients:
        instance.close()


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("CS329Z_LLM_BACKEND", "copilot_bridge")


@pytest.mark.parametrize("cost", [None, 0.125])
def test_default_openrouter_request_and_cost(api, monkeypatch, cost):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-key")
    api.replies.append(chat_reply(cost=cost))
    model = llm.LM()
    assert model(MESSAGES) == "OpenRouter reply"
    request = api.requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer router-test-key"
    assert request.headers["x-title"] == "CS329Z HW1"
    body = json.loads(request.content)
    assert body["messages"] == MESSAGES
    assert body["usage"] == {"include": True}
    assert body["reasoning"] == {"effort": "low"}
    assert body["model"] == "openai/gpt-6-luna"
    assert body["max_tokens"] == 4096
    row, = llm.read_ledger()
    assert row["cost_source"] == ("estimated" if cost is None else "reported")
    assert row["cost_usd"] == (llm.cost_usd(model.model, 30, 0, 10) if cost is None else cost)


def test_openrouter_still_requires_key_and_known_price(api):
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        llm.LM()(MESSAGES)
    with pytest.raises(KeyError, match="No price"):
        llm.LM("unknown-price-model")
    assert not api.requests


def test_invalid_backend_fails_before_network(api, monkeypatch):
    monkeypatch.setenv("CS329Z_LLM_BACKEND", "typo")
    with pytest.raises(ValueError, match="Unknown CS329Z_LLM_BACKEND"):
        llm.LM()
    assert not api.requests


def test_bridge_request_text_usage_and_returned_model(api, bridge, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-sent")
    api.replies.append([
        {"type": "response.output_text.delta", "delta": "OK", "sequence_number": 0,
         "item_id": "msg_test", "output_index": 0, "content_index": 0},
        terminal(),
    ])
    model = llm.LM("not-in-price-table", max_tokens=1234)
    assert model(MESSAGES) == "OK"  # never concatenate the delta and final text
    request = api.requests[0]
    assert str(request.url) == "http://127.0.0.1:8765/codex/responses"
    assert request.headers["authorization"] == "Bearer local-bridge"
    assert "x-title" not in request.headers
    assert json.loads(request.content) == {
        "model": "not-in-price-table", "input": MESSAGES, "stream": True, "store": False,
        "max_output_tokens": 1234, "temperature": 0.0, "reasoning": {"effort": "low"},
    }
    row, = llm.read_ledger()
    assert (row["input_tokens"], row["cached_tokens"], row["output_tokens"], row["reasoning_tokens"]) == (30, 5, 10, 3)
    assert row["cost_usd"] == 0
    assert row["cost_source"] == "unreported"
    assert row["usage_reported"] is True
    assert row["response_model"] == model.usage()["response_model"] == "returned-model"
    assert row["backend"] == "copilot_bridge"
    assert "not a measure of Copilot quota" in llm.spending_report()
    assert model(MESSAGES) == "OK"
    assert model.cached_calls == 1
    assert len(api.requests) == 1
    assert api.responses[0].is_closed


@pytest.mark.parametrize("role,expected", [("grading", "gpt-6-luna"), ("user", "gpt-6-luna"), ("judge", "gpt-6-sol")])
def test_bridge_roles_and_explicit_overrides(bridge, monkeypatch, role, expected):
    assert llm.LM(role).model == expected
    monkeypatch.setenv(f"CS329Z_{role.upper()}_MODEL", "exact/custom-id")
    assert llm.LM(role).model == "exact/custom-id"
    assert llm.LM("openai/explicit-id").model == "openai/explicit-id"


def test_custom_bridge_endpoint_key_and_no_reasoning(api, bridge, monkeypatch):
    monkeypatch.setenv("CS329Z_BRIDGE_BASE_URL", "http://localhost:9000/custom///")
    monkeypatch.setenv("CS329Z_BRIDGE_API_KEY", "bridge-test-key")
    monkeypatch.setenv("CS329Z_REASONING_EFFORT", "")
    api.replies.append([terminal()])
    assert llm.LM()(MESSAGES) == "OK"
    assert str(api.requests[0].url) == "http://localhost:9000/custom/responses"
    assert api.requests[0].headers["authorization"] == "Bearer bridge-test-key"
    assert "reasoning" not in json.loads(api.requests[0].content)


def test_default_directories_are_separate_and_frozen(monkeypatch):
    monkeypatch.delenv("CS329Z_CACHE_DIR")
    router = llm.LM()
    monkeypatch.setenv("CS329Z_LLM_BACKEND", "copilot_bridge")
    bridge_model = llm.LM()
    assert router._cache_directory == llm.REPO_ROOT / ".lm_cache"
    assert bridge_model._cache_directory == llm.REPO_ROOT / ".lm_cache/copilot-bridge"


@pytest.mark.parametrize("usage", [None, {"input_tokens": 9}])
def test_missing_or_partial_usage_is_marked(api, bridge, usage):
    event = terminal()
    event["response"]["usage"] = usage
    api.replies.append([event])
    assert llm.LM()(MESSAGES) == "OK"
    row, = llm.read_ledger()
    assert row["usage_reported"] is False
    assert row["input_tokens"] == (9 if usage else 0)
    assert row["output_tokens"] == 0
    assert "Token usage missing or incomplete" in llm.spending_report()


def test_bridge_reported_cost_is_preserved(api, bridge):
    event = terminal()
    event["response"]["usage"]["cost"] = 0.25
    api.replies.append([event])
    llm.LM()(MESSAGES)
    row, = llm.read_ledger()
    assert row["cost_usd"] == 0.25
    assert row["cost_source"] == "reported"


def test_truncation_records_usage_and_partial_text_but_never_caches(api, bridge, tmp_path):
    api.replies.append([terminal("Partial", status="incomplete", reason="max_output_tokens")])
    model = llm.LM(max_tokens=10)
    with pytest.raises(llm.OutputTruncated) as caught:
        model(MESSAGES)
    assert caught.value.text == "Partial"
    assert caught.value.output_tokens == 10
    assert caught.value.reasoning_tokens == 3
    assert model.output_tokens == 10
    row, = llm.read_ledger()
    assert row["finish_reason"] == "length"
    assert not list(tmp_path.glob("*/*.json"))
    assert api.responses[0].is_closed


@pytest.mark.parametrize("events,match", [
    ([], "without a terminal"),
    ([{"type": "response.output_text.delta", "delta": "partial", "sequence_number": 1,
       "item_id": "msg_test", "output_index": 0, "content_index": 0}], "without a terminal"),
    ([{"type": "error", "code": "server_error", "message": "broken", "sequence_number": 1}], "broken"),
    ([terminal(status="failed")], "failed"),
    ([terminal(status="incomplete", reason="content_filter")], "content_filter"),
])
def test_failed_streams_are_closed_and_never_cached(api, bridge, tmp_path, events, match):
    api.replies.append(events)
    with pytest.raises(RuntimeError, match=match):
        llm.LM()(MESSAGES)
    assert not list(tmp_path.glob("*/*.json"))
    assert not llm.read_ledger()
    assert api.responses[0].is_closed
    assert len(api.requests) == 1


def test_text_protocol_rejects_native_tools(api, bridge):
    event = terminal()
    event["response"]["output"] = [{
        "type": "function_call", "id": "fc_test", "call_id": "call_test",
        "name": "run", "arguments": "{}", "status": "completed",
    }]
    api.replies.append([event])
    with pytest.raises(RuntimeError, match="expects text only"):
        llm.LM()(MESSAGES)


def test_refusal_is_returned_as_text_and_reasoning_is_not(api, bridge):
    event = terminal()
    event["response"]["output"][0]["content"] = [{"type": "refusal", "refusal": "Cannot do that."}]
    event["response"]["output"].insert(0, {"type": "reasoning", "id": "rs_test", "summary": []})
    api.replies.append([event])
    assert llm.LM()(MESSAGES) == "Cannot do that."


def test_empty_reply_is_not_cached(api, bridge):
    api.replies.extend([[terminal("")], [terminal("Second")]])
    model = llm.LM()
    assert model(MESSAGES) == ""
    assert model(MESSAGES) == "Second"
    assert len(api.requests) == 2


def test_rejected_parameters_retry_once_each_and_are_remembered(api, bridge):
    api.replies.extend([
        rejection("temperature"), rejection("reasoning.effort"), [terminal()], [terminal()],
    ])
    assert llm.LM(cache=False)(MESSAGES) == "OK"
    assert llm.LM(cache=False)(MESSAGES) == "OK"
    bodies = [json.loads(r.content) for r in api.requests]
    assert "temperature" in bodies[0] and "reasoning" in bodies[0]
    assert "temperature" not in bodies[1] and "reasoning" in bodies[1]
    assert all("temperature" not in b and "reasoning" not in b for b in bodies[2:])
    assert all(b["max_output_tokens"] == 4096 for b in bodies)


@pytest.mark.parametrize("event_type", ["error", "response.failed"])
def test_sse_parameter_rejection_can_retry_before_output(api, bridge, event_type):
    error = {"code": "unsupported_parameter", "param": "temperature", "message": "Temperature is not supported"}
    if event_type == "error":
        event = {"type": "error", "sequence_number": 1, **error}
    else:
        event = terminal(status="failed")
        event["response"]["error"] = error
    api.replies.extend([[event], [terminal()]])
    assert llm.LM()(MESSAGES) == "OK"
    assert "temperature" not in json.loads(api.requests[1].content)
    assert all(r.is_closed for r in api.responses)


@pytest.mark.parametrize("parameter,status", [("temperature", 401), ("temperature", 500), ("max_output_tokens", 400)])
def test_auth_server_and_output_limit_errors_do_not_change_request(api, bridge, parameter, status):
    api.replies.append(rejection(parameter, status=status))
    with pytest.raises(openai.APIStatusError):
        llm.LM()(MESSAGES)
    assert len(api.requests) == 1


def test_parameter_name_in_unrelated_error_is_not_a_retry_signal(api, bridge):
    api.replies.append(httpx.Response(400, json={"error": {
        "message": "Server failed while processing temperature", "code": "server_error",
    }}))
    with pytest.raises(openai.BadRequestError):
        llm.LM()(MESSAGES)
    assert len(api.requests) == 1


def test_repeated_rejection_is_bounded(api, bridge):
    api.replies.extend([rejection("temperature"), rejection("temperature")])
    with pytest.raises(openai.BadRequestError):
        llm.LM()(MESSAGES)
    assert len(api.requests) == 2


def test_no_retry_after_partial_output(api, bridge):
    api.replies.append([
        {"type": "response.output_text.delta", "delta": "Partial", "sequence_number": 1,
         "item_id": "msg_test", "output_index": 0, "content_index": 0},
        {"type": "error", "code": "unsupported_parameter", "param": "temperature",
         "message": "Unsupported temperature", "sequence_number": 2},
    ])
    with pytest.raises(RuntimeError, match="Unsupported temperature"):
        llm.LM()(MESSAGES)
    assert len(api.requests) == 1


def test_no_retry_after_terminal_output(api, bridge):
    api.replies.append([
        terminal(),
        {"type": "error", "code": "unsupported_parameter", "param": "temperature",
         "message": "Unsupported temperature", "sequence_number": 2},
    ])
    with pytest.raises(RuntimeError, match="Unsupported temperature"):
        llm.LM()(MESSAGES)
    assert len(api.requests) == 1


def test_mismatched_terminal_status_is_not_success(api, bridge):
    event = terminal()
    event["response"]["status"] = "in_progress"
    api.replies.append([event])
    with pytest.raises(RuntimeError, match="terminal status"):
        llm.LM()(MESSAGES)


def test_endpoint_isolation_for_cache_and_parameter_learning(api, bridge, monkeypatch):
    api.replies.extend([rejection("temperature"), [terminal("First")], [terminal("Second")]])
    first = llm.LM()
    assert first(MESSAGES) == "First"
    monkeypatch.setenv("CS329Z_BRIDGE_BASE_URL", "http://localhost:9999/codex")
    second = llm.LM()
    assert second(MESSAGES) == "Second"
    assert "temperature" in json.loads(api.requests[-1].content)
    assert first(MESSAGES) == "First"
    assert second(MESSAGES) == "Second"
    assert len(api.requests) == 3


def test_provider_isolation_with_same_model_and_directory(api, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-key")
    api.replies.extend([rejection("temperature"), chat_reply(), [terminal("Bridge")]])
    router = llm.LM("gpt-6-luna")
    assert router(MESSAGES) == "OpenRouter reply"
    monkeypatch.setenv("CS329Z_LLM_BACKEND", "copilot_bridge")
    bridge_model = llm.LM("gpt-6-luna")
    assert bridge_model(MESSAGES) == "Bridge"
    assert "temperature" in json.loads(api.requests[-1].content)
    assert router(MESSAGES) == "OpenRouter reply"
    assert len(api.requests) == 3


def test_budget_blocks_network_but_allows_cached_reply(api, bridge, monkeypatch):
    api.replies.append([terminal()])
    model = llm.LM()
    assert model(MESSAGES) == "OK"
    monkeypatch.setenv("CS329Z_BUDGET_USD", "0")
    assert model(MESSAGES) == "OK"
    with pytest.raises(llm.BudgetExceeded):
        llm.LM(cache=False)(MESSAGES)
    assert len(api.requests) == 1


def test_old_sdk_error_is_actionable(bridge):
    model = llm.LM()
    model._client = SimpleNamespace()
    with pytest.raises(RuntimeError, match="uv sync --locked"):
        model(MESSAGES)
