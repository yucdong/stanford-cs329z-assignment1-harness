"""Course-provided language model wrapper.

This module gives you exactly one primitive: a model call that takes
a list of messages and returns a string.

    lm = LM()                                  # the grading model
    text = lm([{"role": "system", "content": "..."},
               {"role": "user", "content": "..."}])

There is no tool-calling parameter and no structured-output parameter. Those
are yours to build on top of this call.

What the wrapper does for you:

* Disk cache. Every call is cached under ``.lm_cache/`` keyed on the backend,
  endpoint, model, sampling settings, and exact messages. Re-running unchanged
  code costs nothing.
* Cost ledger. Every call that reaches the provider is appended to
  ``.lm_cache/ledger.jsonl`` with its token counts and dollar cost.
  ``uv run python -m cs329z_hw1.llm`` prints a spending report.
* Budget stop. If ``CS329Z_BUDGET_USD`` is set, the wrapper raises
  ``BudgetExceeded`` instead of making a call once the ledger total passes it.
* Truncation. A call that stops because the model reached ``max_tokens``
  raises ``OutputTruncated`` instead of returning the cut-off text. The
  exception carries the text produced so far (``.text``) and the token
  counts. The call is charged and recorded in the ledger but not cached.

``ScriptedLM`` is a drop-in fake used by the deterministic tests. It replays a
fixed list of replies and never touches the network.

OpenRouter is the default backend. ``CS329Z_LLM_BACKEND=copilot_bridge`` uses
the bridge's streaming Responses endpoint and defaults to a separate cache
and ledger under ``.lm_cache/copilot-bridge``. Unreported bridge costs are
recorded as zero with ``cost_source="unreported"``, not estimated charges.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}

# Model used for each role. Override any of these with the environment
# variable CS329Z_<ROLE>_MODEL, e.g. CS329Z_GRADING_MODEL.
DEFAULT_MODELS = {
    "grading": "openai/gpt-6-luna",  # the model your agent is graded with
    "user": "openai/gpt-6-luna",  # plays the simulated users
    "judge": "openai/gpt-6-sol",  # scores evaluation transcripts
}

# By default, calls go to OpenRouter, which fronts many providers behind one
# OpenAI-compatible API. Model names carry the provider as a prefix
# ("openai/gpt-6-luna"); https://openrouter.ai/models lists them.
OPENROUTER_URL = "https://openrouter.ai/api/v1"
BRIDGE_URL = "http://127.0.0.1:8765/codex"

# Reasoning effort requested from models that accept it (GPT-5 and GPT-6).
# Override with CS329Z_REASONING_EFFORT.
DEFAULT_REASONING_EFFORT = "low"

# USD per 1M tokens: (input, cached input, output), keyed by the model name
# without its provider prefix. OpenRouter reports the exact cost of each
# call in the response, and the ledger records that; this table is the
# fallback for an OpenRouter response without one, and it is checked when an
# OpenRouter LM is created so an unknown model fails before a paid call. Read from
# https://openrouter.ai/models on 2026-10-05 (the same as OpenAI's list
# prices).
PRICES = {
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6-sol": (2.00, 0.20, 10.00),
}

VALID_ROLES = ("system", "user", "assistant")


class BudgetExceeded(RuntimeError):
    """Raised instead of calling the provider once the spend cap is reached."""


class ScriptExhausted(RuntimeError):
    """Raised when a ScriptedLM is called more times than its script allows."""


class OutputTruncated(RuntimeError):
    """Raised when the model stopped because it reached ``max_tokens``.

    ``text`` is what the model wrote before the cut, which is often empty:
    a reasoning model's hidden reasoning counts against the same limit, so
    the limit can be used up before any visible text is written.
    ``output_tokens`` is the whole output, ``reasoning_tokens`` the hidden
    part of it (0 when the provider does not report it). Calling again with
    a larger ``max_tokens`` is a new paid call.
    """

    def __init__(self, text: str, *, model: str, max_tokens: int, output_tokens: int, reasoning_tokens: int = 0):
        self.text = text
        self.model = model
        self.max_tokens = max_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        shown = f"{len(text)} characters of text" if text else "no visible text"
        super().__init__(
            f"{model} stopped at max_tokens={max_tokens} with {shown} "
            f"({output_tokens} output tokens, {reasoning_tokens} of them hidden reasoning). "
            "Ask for a shorter reply, or call again with a larger max_tokens."
        )


def resolve_model(role_or_model: str | None = None) -> str:
    """Map a role ("grading", "user", "judge") to a model name.

    ``None`` means "grading". A string that is not a role is returned
    unchanged, so you can pass a model name.
    """
    name = role_or_model or "grading"
    if name in DEFAULT_MODELS:
        default = DEFAULT_MODELS[name]
        if _backend() == "copilot_bridge":
            default = default.removeprefix("openai/")
        return os.environ.get(f"CS329Z_{name.upper()}_MODEL", default)
    return name


def _price(model: str) -> tuple[float, float, float]:
    model = model.rsplit("/", 1)[-1]  # "openai/gpt-6-luna" -> "gpt-6-luna"
    best = None
    for prefix, price in PRICES.items():
        if model == prefix or model.startswith(prefix + "-20"):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, price)
    if best is None:
        raise KeyError(
            f"No price on file for model {model!r}. Add it to PRICES in cs329z_hw1/llm.py."
        )
    return best[1]


def cost_usd(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    p_in, p_cached, p_out = _price(model)
    fresh = max(input_tokens - cached_tokens, 0)
    return (fresh * p_in + cached_tokens * p_cached + output_tokens * p_out) / 1_000_000


def _backend() -> str:
    backend = os.environ.get("CS329Z_LLM_BACKEND", "openrouter")
    if backend not in {"openrouter", "copilot_bridge"}:
        raise ValueError(
            f"Unknown CS329Z_LLM_BACKEND {backend!r}; use 'openrouter' or 'copilot_bridge'."
        )
    return backend


def _cache_dir() -> Path:
    default = REPO_ROOT / ".lm_cache"
    if _backend() == "copilot_bridge":
        default /= "copilot-bridge"
    return Path(os.environ.get("CS329Z_CACHE_DIR", default))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(REPO_ROOT / ".env")


_ledger_lock = threading.Lock()
# Models that reject the temperature or reasoning_effort parameters, learned
# from the provider's error the first time each is used.
_NO_TEMPERATURE: set[tuple[str, ...]] = set()
_NO_EFFORT: set[tuple[str, ...]] = set()


def read_ledger(cache_dir: Path | None = None) -> list[dict]:
    path = (cache_dir if cache_dir is not None else _cache_dir()) / "ledger.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def total_spend(cache_dir: Path | None = None) -> float:
    return sum(row.get("cost_usd", 0.0) for row in read_ledger(cache_dir))


def _append_ledger(row: dict, cache_dir: Path | None = None) -> None:
    path = (cache_dir if cache_dir is not None else _cache_dir()) / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _validate_messages(messages: Sequence[Message]) -> None:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a non-empty list of {'role', 'content'} dicts")
    for m in messages:
        if not isinstance(m, dict) or set(m) != {"role", "content"}:
            raise ValueError(
                f"each message must be a dict with exactly the keys 'role' and 'content', got {m!r}"
            )
        if m["role"] not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {m['role']!r}")
        if not isinstance(m["content"], str):
            raise ValueError("message content must be a string")


class LM:
    """A chat model: messages in, string out.

    Parameters
    ----------
    model:
        A role ("grading", "user", "judge"), a model ID accepted by the
        selected backend, or ``None`` for "grading".
    temperature, max_tokens:
        Sampling settings. They are part of the cache key. Reasoning models
        spend part of ``max_tokens`` on hidden reasoning, so keep it large.
        Models that reject ``temperature`` are called without it.
    tag:
        A free-form label written to the cost ledger so spending can be
        grouped (the test suite sets it to the test name).
    cache:
        Set to False to bypass the disk cache for this instance.
    salt:
        Extra string mixed into the cache key. Two instances with different
        salts get independent samples for identical messages.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tag: str | None = None,
        cache: bool = True,
        salt: str = "",
    ) -> None:
        _load_env()
        self.backend = _backend()
        self.api_type = "responses" if self.backend == "copilot_bridge" else "chat_completions"
        self.base_url = (
            (os.environ.get("CS329Z_BRIDGE_BASE_URL") or BRIDGE_URL).rstrip("/")
            if self.backend == "copilot_bridge" else OPENROUTER_URL
        )
        self.model = resolve_model(model)
        if self.backend == "openrouter":
            _price(self.model)  # fail before a paid call if the price is unknown
        self._endpoint_key = (self.backend, self.base_url, self.api_type, self.model)
        self._cache_directory = _cache_dir()
        self.reasoning_effort = os.environ.get("CS329Z_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        self.response_model: str | None = None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tag = tag
        self.cache = cache
        self.salt = salt
        # Running totals for this instance.
        self.calls = 0
        self.cached_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.seconds = 0.0
        self._client = None

    # -- public API -------------------------------------------------------

    def __call__(self, messages: Sequence[Message]) -> str:
        _validate_messages(messages)
        messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        key = self._key(messages)
        path = self._cache_directory / key[:2] / f"{key}.json"
        self.calls += 1
        if self.cache and path.exists():
            self.cached_calls += 1
            cached = json.loads(path.read_text(encoding="utf-8"))
            self.response_model = cached.get("response_model", self.model)
            return cached["text"]

        self._check_budget()
        start = time.time()
        text, usage = self._complete(messages)
        elapsed = time.time() - start
        self.response_model = usage.get("response_model", self.model)
        cost = usage.get("cost")
        cost_source = "reported"
        if cost is None:
            if self.backend == "openrouter":
                cost = cost_usd(self.model, usage["input"], usage["cached"], usage["output"])
                cost_source = "estimated"
            else:
                cost = 0.0
                cost_source = "unreported"
        self.input_tokens += usage["input"]
        self.output_tokens += usage["output"]
        self.cost_usd += cost
        self.seconds += elapsed
        truncated = usage.get("finish_reason") == "length"
        _append_ledger(
            {
                "ts": round(start, 3),
                "model": self.model,
                "response_model": self.response_model,
                "backend": self.backend,
                "base_url": self.base_url,
                "api_type": self.api_type,
                "tag": self.tag or os.environ.get("CS329Z_TAG", ""),
                "input_tokens": usage["input"],
                "cached_tokens": usage["cached"],
                "output_tokens": usage["output"],
                "reasoning_tokens": usage.get("reasoning", 0),
                "usage_reported": usage.get("usage_reported", True),
                "cost_usd": cost,
                "cost_source": cost_source,
                "seconds": round(elapsed, 3),
                "finish_reason": usage.get("finish_reason"),
            },
            self._cache_directory,
        )
        if truncated:  # charged and logged, never cached
            raise OutputTruncated(
                text,
                model=self.model,
                max_tokens=self.max_tokens,
                output_tokens=usage["output"],
                reasoning_tokens=usage.get("reasoning", 0),
            )
        if self.cache and text.strip():  # an empty reply is not worth remembering
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(
                json.dumps({"text": text, "model": self.model, "response_model": self.response_model}),
                encoding="utf-8",
            )
            tmp.replace(path)
        return text

    def usage(self) -> dict:
        """Totals for this instance since it was created."""
        return {
            "model": self.model,
            "backend": self.backend,
            "response_model": self.response_model,
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "seconds": self.seconds,
        }

    # -- internals --------------------------------------------------------

    def _key(self, messages: list[Message]) -> str:
        blob = json.dumps(
            [self._endpoint_key, self.temperature, self.max_tokens, self.reasoning_effort, self.salt, messages],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _check_budget(self) -> None:
        cap = os.environ.get("CS329Z_BUDGET_USD")
        if not cap:
            return
        spent = total_spend(self._cache_directory)
        if spent >= float(cap):
            raise BudgetExceeded(
                f"Spent ${spent:.2f} of the ${float(cap):.2f} cap set by CS329Z_BUDGET_USD. "
                "Raise the cap in .env if you mean to continue."
            )

    def _complete(self, messages: list[Message]) -> tuple[str, dict]:
        if self._client is None:
            from openai import OpenAI

            key = (
                os.environ.get("CS329Z_BRIDGE_API_KEY") or "local-bridge"
                if self.backend == "copilot_bridge" else os.environ.get("OPENROUTER_API_KEY")
            )
            if not key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=key,
                max_retries=6,
                timeout=120,
                default_headers={"X-Title": "CS329Z HW1"} if self.backend == "openrouter" else {},
            )
        if self.backend == "copilot_bridge":
            return self._complete_responses(messages)
        return self._complete_openrouter(messages)

    def _complete_openrouter(self, messages: list[Message]) -> tuple[str, dict]:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            # OpenRouter extensions: the reasoning effort, and the request
            # that the response carry the cost of the call.
            "extra_body": {"usage": {"include": True}},
        }
        if self._endpoint_key not in _NO_TEMPERATURE:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort and self._endpoint_key not in _NO_EFFORT:
            kwargs["extra_body"]["reasoning"] = {"effort": self.reasoning_effort}
        response = self._request_with_fallback(self._client.chat.completions.create, kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = _normalize_usage(response.usage, responses=False)
        usage.update(finish_reason=choice.finish_reason, response_model=response.model)
        return text, usage

    def _complete_responses(self, messages: list[Message]) -> tuple[str, dict]:
        if not hasattr(self._client, "responses"):
            raise RuntimeError(
                "The installed OpenAI SDK does not support Responses. "
                "Run 'uv sync --locked' and use 'uv run --locked' for this project."
            )
        kwargs: dict = {
            "model": self.model,
            "input": messages,
            "stream": True,
            "store": False,
            "max_output_tokens": self.max_tokens,
        }
        if self._endpoint_key not in _NO_TEMPERATURE:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort and self._endpoint_key not in _NO_EFFORT:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = self._request_with_fallback(self._read_response, kwargs)
        parts = []
        for item in response.output:
            if item.type == "reasoning":
                continue
            if item.type != "message":
                raise RuntimeError(f"Unexpected Responses output {item.type!r}; LM expects text only.")
            for part in item.content:
                if part.type == "output_text":
                    parts.append(part.text)
                elif part.type == "refusal":
                    parts.append(part.refusal)
                else:
                    raise RuntimeError(f"Unexpected Responses content {part.type!r}.")
        usage = _normalize_usage(response.usage, responses=True)
        usage.update(
            finish_reason="length" if response.status == "incomplete" else "stop",
            response_model=response.model,
        )
        return "".join(parts), usage

    def _read_response(self, **kwargs):
        terminal = None
        output_seen = False
        # Errors can arrive during iteration, not just during create().
        with self._client.responses.create(**kwargs) as stream:
            for event in stream:
                if event.type == "error":
                    raise _ResponsesError(event.model_dump(), retryable=not output_seen)
                if event.type == "response.failed":
                    error = event.response.error
                    raise _ResponsesError(
                        error.model_dump() if error else {"message": "Responses request failed."},
                        retryable=not output_seen,
                    )
                if event.type in {"response.completed", "response.incomplete"}:
                    terminal = event.response
                    output_seen = True
                    if event.type != f"response.{terminal.status}":
                        raise RuntimeError(f"Unexpected Responses terminal status: {terminal.status!r}.")
                    if terminal.status == "incomplete":
                        reason = getattr(terminal.incomplete_details, "reason", None)
                        if reason != "max_output_tokens":
                            raise RuntimeError(f"Responses request incomplete: {reason!r}.")
                elif event.type.startswith("response.output_") or event.type.endswith(".delta"):
                    output_seen = True
        if terminal is None:
            raise RuntimeError("Responses stream ended without a terminal response.")
        return terminal

    def _request_with_fallback(self, request: Callable, kwargs: dict):
        # Each removal can happen only once. Never drop the output limit or
        # retry a stream that has already started producing output.
        while True:
            try:
                return request(**kwargs)
            except Exception as exc:
                parameter = _unsupported_parameter(exc)
                reasoning_options = kwargs.get("extra_body", kwargs)
                if parameter == "temperature" and "temperature" in kwargs:
                    _NO_TEMPERATURE.add(self._endpoint_key)
                    kwargs.pop("temperature")
                elif parameter == "reasoning" and "reasoning" in reasoning_options:
                    _NO_EFFORT.add(self._endpoint_key)
                    reasoning_options.pop("reasoning")
                else:
                    raise


class _ResponsesError(RuntimeError):
    def __init__(self, body: dict, *, retryable: bool):
        self.body = body
        self.retryable = retryable
        super().__init__(f"Responses stream failed: {body.get('message') or body.get('code') or 'unknown error'}")


def _unsupported_parameter(exc: Exception) -> str | None:
    """Only explicit parameter rejections may change a request and retry."""
    from openai import APIStatusError

    if isinstance(exc, _ResponsesError):
        if not exc.retryable:
            return None
    elif not isinstance(exc, APIStatusError) or exc.status_code not in {400, 422}:
        return None
    body = exc.body
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    message = str(error.get("message", "")).lower()
    if code not in {"unsupported_parameter", "unsupported_value"} and not any(
        phrase in message for phrase in ("not supported", "unsupported", "does not support", "only the default")
    ):
        return None
    parameter = str(error.get("param") or "").split(".", 1)[0]
    if parameter in {"temperature", "reasoning", "reasoning_effort"}:
        return "reasoning" if parameter.startswith("reasoning") else parameter
    if parameter:  # An explicit rejection of some other field is not ours to fix.
        return None
    mentioned = [p for p in ("temperature", "reasoning") if re.search(rf"\b{p}(?:_effort)?\b", message)]
    return mentioned[0] if len(mentioned) == 1 else None


def _normalize_usage(u, *, responses: bool) -> dict:
    input_field, output_field = ("input_tokens", "output_tokens") if responses else ("prompt_tokens", "completion_tokens")
    input_tokens = getattr(u, input_field, None)
    output_tokens = getattr(u, output_field, None)
    details = getattr(u, f"{input_field}_details", None)
    out_details = getattr(u, f"{output_field}_details", None)
    usage = {
        "input": input_tokens or 0,
        "cached": getattr(details, "cached_tokens", 0) or 0,
        "output": output_tokens or 0,
        "reasoning": getattr(out_details, "reasoning_tokens", 0) or 0,
        "usage_reported": input_tokens is not None and output_tokens is not None,
    }
    cost = getattr(u, "cost", None)
    if isinstance(cost, (int, float)):
        usage["cost"] = float(cost)
    return usage


class ScriptedLM:
    """A fake LM that replays a fixed script. Used by the deterministic tests.

    Each script entry is one model reply, given as one of:

    * a string, returned as is;
    * a dict ``{"text": str, "tool_call": {"name": str, "args": dict}}``,
      rendered to a string by ``formatter(text, ToolCall)``. The tests pass
      your ``run_format_tool_call`` adapter as the formatter, so scripted
      tool calls arrive in your own format;
    * a callable ``f(messages) -> str``.

    Every call's messages are recorded in ``.calls`` so tests can inspect
    exactly what your code sent to the model. When the script runs out the
    LM raises ``ScriptExhausted``, unless ``repeat_last=True``, in which case
    the last entry is replayed forever.
    """

    model = "scripted"

    def __init__(
        self,
        script: Sequence[Any],
        *,
        formatter: Callable[[str, Any], str] | None = None,
        repeat_last: bool = False,
    ) -> None:
        self.script = list(script)
        self.formatter = formatter
        self.repeat_last = repeat_last
        self.calls: list[list[Message]] = []
        self.replies: list[str] = []

    def __call__(self, messages: Sequence[Message]) -> str:
        _validate_messages(messages)
        index = len(self.calls)
        self.calls.append(copy.deepcopy(list(messages)))
        if index >= len(self.script):
            if not (self.repeat_last and self.script):
                raise ScriptExhausted(
                    f"ScriptedLM was called {index + 1} times but the script has "
                    f"{len(self.script)} entries. Your code made a model call the test "
                    "did not expect."
                )
            index = len(self.script) - 1
        entry = self.script[index]
        if callable(entry):
            reply = entry(self.calls[-1])
        elif isinstance(entry, dict):
            reply = self._render(entry)
        else:
            reply = str(entry)
        self.replies.append(reply)
        return reply

    @property
    def remaining(self) -> int:
        return max(len(self.script) - len(self.calls), 0)

    def _render(self, entry: dict) -> str:
        text = entry.get("text", "")
        call = entry.get("tool_call")
        if call is None:
            return text
        if self.formatter is None:
            raise RuntimeError("ScriptedLM needs a formatter to render scripted tool calls")
        from cs329z_hw1.types import ToolCall

        return self.formatter(text, ToolCall(name=call["name"], args=dict(call.get("args", {}))))

    def usage(self) -> dict:
        return {"model": "scripted", "calls": len(self.calls), "cost_usd": 0.0}


def spending_report(rows: list[dict] | None = None) -> str:
    rows = read_ledger() if rows is None else rows
    if not rows:
        return "No LLM calls recorded yet."
    by_model: dict[str, dict] = {}
    by_tag: dict[str, dict] = {}
    for row in rows:
        model = row.get("model", "?")
        short = model.split("-20")[0]  # drop the snapshot date
        tag_key = f"{row.get('tag') or '(untagged)'} [{short}]"
        for table, key in ((by_model, model), (by_tag, tag_key)):
            agg = table.setdefault(key, {"calls": 0, "cut": 0, "in": 0, "out": 0, "usd": 0.0, "sec": 0.0})
            agg["calls"] += 1
            agg["cut"] += row.get("finish_reason") == "length"
            agg["in"] += row.get("input_tokens", 0)
            agg["out"] += row.get("output_tokens", 0)
            agg["usd"] += row.get("cost_usd", 0.0)
            agg["sec"] += row.get("seconds", 0.0)
    lines = []
    for title, table in (("By model", by_model), ("By tag and model", by_tag)):
        lines.append(title)
        lines.append(f"  {'name':<72}{'calls':>7}{'cut':>5}{'in tok':>12}{'out tok':>10}{'USD':>9}{'sec':>8}")
        for key, agg in sorted(table.items(), key=lambda kv: -kv[1]["usd"]):
            lines.append(
                f"  {key[-71:]:<72}{agg['calls']:>7}{agg['cut'] or '':>5}{agg['in']:>12}{agg['out']:>10}"
                f"{agg['usd']:>9.3f}{agg['sec']:>8.0f}"
            )
        lines.append("")
    total = sum(r.get("cost_usd", 0.0) for r in rows)
    cut = sum(r.get("finish_reason") == "length" for r in rows)
    lines.append(f"Total: {len(rows)} calls, ${total:.3f}" + (f"; {cut} cut off at max_tokens" if cut else ""))
    unpriced = sum(r.get("cost_source") == "unreported" for r in rows)
    uncounted = sum(r.get("usage_reported") is False for r in rows)
    if unpriced:
        lines.append(f"Cost unreported for {unpriced} calls (recorded as $0; not a measure of Copilot quota or charges).")
    if uncounted:
        lines.append(f"Token usage missing or incomplete for {uncounted} calls (missing counts recorded as 0).")
    return "\n".join(lines)


if __name__ == "__main__":  # uv run python -m cs329z_hw1.llm
    _load_env()
    print(spending_report())
