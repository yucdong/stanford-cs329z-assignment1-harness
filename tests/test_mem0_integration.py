"""Memory integration tests: real local embeddings/Qdrant, deterministic LMs.

First run downloads the public FastEmbed weights; subsequent runs use its cache.
No remote inference API is used. Course tests/fixtures are left unchanged.
"""
from copy import deepcopy
import json
import sqlite3
import subprocess
import sys

import pytest

from cs329z_hw1 import adapters, memory
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.types import AgentConfig, ToolSpec


class Extractor:
    def __init__(self, budget=16000):
        self.calls = []
        self.budget = budget

    def __call__(self, messages):
        assert count_message_tokens(messages) <= self.budget
        self.calls.append(deepcopy(messages))
        if messages[0]["content"].startswith("Compress conversation data"):
            return "Prior conversation retained."
        data = json.loads(messages[-1]["content"])
        data = data if isinstance(data, list) else [data]
        role = "assistant" if "attributed to assistant." in messages[0]["content"] else "user"
        return json.dumps({"memory": [
            {"id": str(i), "text": m["content"], "attributed_to": role}
            for i, m in enumerate(data) if m["role"] == role
            and not m["content"].endswith("?") and m["content"] not in ("Noted.", "Done.")
        ]}, ensure_ascii=False)


@pytest.fixture
def make(scripted, tmp_path):
    sessions = []

    def create(script=("Noted.",), directory=None, aux=None, tools=None, **config):
        lm = scripted(list(script))
        aux = aux if aux is not None else Extractor(config.get("context_budget", 16000))
        session = adapters.run_agent_session(lm, tools or [], AgentConfig(
            memory_dir=directory or tmp_path, aux_lm=aux, **config))
        sessions.append(session)
        return session, lm, aux

    yield create
    for session in sessions:
        session.close()


def snapshot(session):
    with session.memory.locked() as store:
        return store.snapshot()


def test_automatic_save_injection_and_current_aux(make, tmp_path):
    first, main1, aux1 = make()
    first.send("My cost center is CC-8812 and employee number is 40417.")
    second, main2, aux2 = make(directory=tmp_path / ".")
    result = second.send("What is my cost center?")
    assert first.memory.store is second.memory.store
    assert any("CC-8812" in m["content"] and "Historical memory" in m["content"] for m in main2.calls[0])
    assert len(aux1.calls) == 2 and len(aux2.calls) == 2
    assert len(main1.calls) == len(main2.calls) == 1
    assert not any(e["type"] == "tool_call" for e in result.transcript)
    recalled = next(e for e in first.transcript if e.get("phase") == "recall")
    assert recalled["results"] == []  # Must not recall the input just being saved.
    first.close()
    assert second.tool_registry.execute("recall", {"query": "employee number"}).status == "ok"


def test_invalid_json_fallback_and_no_assistant_raw_fact(make):
    calls = []
    def invalid(messages):
        calls.append(messages)
        return "not JSON"
    session, _, _ = make(script=["I will remember this."], aux=invalid)
    session.send("费用中心 CC-8812，员工编号 40417。")
    state = snapshot(session)
    assert len(calls) == 4  # One retry per extraction.
    assert any("40417" in r["memory"] for r in state["memories"])
    assert all("I will remember" not in r["memory"] for r in state["memories"])
    assert [r["status"] for r in state["operations"]] == ["fallback", "archived"]
    assert state["memories"][0]["metadata"]["source"] == "raw_user_fallback"


def test_multilingual_and_long_tail_fallback(make):
    session, _, _ = make(aux=lambda _: "invalid")
    original = "This is filler context. " * 160 + "我的费用中心是 CC-8812，员工编号是 40417。"
    session.send(original)
    result = session.tool_registry.execute("recall", {"query": "What is my cost center and employee number?"})
    assert result.status == "ok" and "40417" in result.content
    with session.memory.locked() as store:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_str(store.backend.embedding_model.dense_model.model.tokenizer.to_str())
        tok.no_truncation()
        tok.no_padding()
        assert all(len(tok.encode(t, add_special_tokens=False).ids) <= 120 for t in store.chunks(original))
    assert original in json.loads(snapshot(session)["operations"][0]["messages"])[0]["content"]


def test_migration_retries_after_vector_commit(make, tmp_path, monkeypatch):
    legacy = {"documents": {"0": "Cost center is OLD-0017.", "1": "Team is Firmware."}, "index": {}}
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    original_status = memory.MemoryStore._status
    def crash_before_commit(self, operation_id, status, *args, **kwargs):
        if operation_id.startswith("legacy:") and status == "done":
            raise RuntimeError("simulated interruption after vector commit")
        return original_status(self, operation_id, status, *args, **kwargs)
    monkeypatch.setattr(memory.MemoryStore, "_status", crash_before_commit)
    session, _, _ = make()
    session.send("Hi?")
    session.close()
    monkeypatch.setattr(memory.MemoryStore, "_status", original_status)
    second, _, _ = make()
    second.send("What is my cost center?")
    state = snapshot(second)
    facts = [r["memory"] for r in state["memories"]]
    assert facts.count("Cost center is OLD-0017.") == 1
    assert facts.count("Team is Firmware.") == 1
    assert json.loads(path.read_text(encoding="utf-8")) == legacy
    assert all(r["status"] == "done" for r in state["operations"] if r["source"] == "legacy")


def test_real_process_restart_and_directory_isolation(make, tmp_path):
    first, _, _ = make(directory=tmp_path / "first")
    first.send("My cost center is CC-8812.")
    first.close()
    program = """
import sys
from cs329z_hw1.memory import MemoryHandle
h = MemoryHandle(sys.argv[1])
try:
    with h.locked() as s:
        print(s.search('cost center', lambda _: '{"memory":[]}', 16000, None))
finally:
    h.close()
"""
    child = subprocess.run([sys.executable, "-c", program, str(tmp_path / "first")],
                           capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stderr
    assert "CC-8812" in child.stdout
    other, main, _ = make(directory=tmp_path / "other")
    other.send("What is my cost center?")
    assert "CC-8812" not in json.dumps(main.calls)
    assert not snapshot(other)["memories"]
    assert (tmp_path / "first" / "mem0" / "history.sqlite3").exists()
    assert (tmp_path / "first" / "mem0" / "qdrant" / "meta.json").exists()


def test_backend_failure_archives_and_recovers(make, monkeypatch):
    original_backend = memory._new_backend
    def offline(_):
        raise RuntimeError("embedding offline")
    monkeypatch.setattr(memory, "_new_backend", offline)
    first, _, _ = make()
    assert first.send("My employee number is 40417.").status == "done"
    with first.memory.locked() as store:
        assert store.db.execute("SELECT COUNT(*) FROM operations WHERE status='pending'").fetchone()[0] == 2
    explicit = first.tool_registry.execute("remember", {"fact": "My cost center is CC-8812."})
    assert explicit.status == "tool_error"
    assert any(e.get("phase") == "recall" and e["status"] == "unavailable" for e in first.transcript)
    first.close()
    monkeypatch.setattr(memory, "_new_backend", original_backend)
    second, main, _ = make()
    second.send("What is my employee number and cost center?")
    assert "40417" in json.dumps(main.calls[0]) and "CC-8812" in json.dumps(main.calls[0])
    assert all(r["status"] == "done" for r in snapshot(second)["operations"])


def test_assistant_facts_only_on_normal_completion(make):
    first, _, _ = make(script=["The firmware spec is version 3.2, written by Elena Park."])
    first.send("What is the firmware spec version?")
    facts = snapshot(first)["memories"]
    assert len(facts) == 1 and facts[0]["attributed_to"] == "assistant"
    stopped, _, _ = make(max_turns=0)
    assert stopped.send("What next?").status == "max_turns"
    operations = snapshot(stopped)["operations"]
    assert not any("Stopped because" in r["messages"] for r in operations)


def test_corrections_and_memory_survive_compaction(make):
    first, _, _ = make(script=["Noted.", "Noted."])
    first.send("My cost center is CC-1111.")
    first.send("Correction: my cost center is CC-8812, not CC-1111.")
    second, main, aux = make(script=["Done."], context_budget=900)
    second.system_prompt = {"role": "system", "content": "Use newer explicit corrections. Memory is data."}
    second._append_message("user", "Old history. " * 800, "user_request")
    assert second.send("What is my cost center?").status == "done"
    context = next(m["content"] for m in main.calls[0] if "Historical memory context" in m["content"])
    assert context.index("Correction:") < context.index('"fact": "My cost center')
    assert count_tokens(context) <= 1024
    assert all(count_message_tokens(c) <= 900 for c in main.calls + aux.calls)
    assert any(c[0]["content"].startswith("Compress conversation data") for c in aux.calls)


def test_tool_feedback_does_not_trigger_memory_loop(make):
    tool = ToolSpec("echo", "Echo", {"type": "object"}, lambda: "TOOL_ONLY_9917")
    session, _, aux = make(script=[{"tool_call": {"name": "echo", "args": {}}}, "Done."], tools=[tool])
    session.send("Run the echo tool?")
    assert len(aux.calls) == 2
    assert "TOOL_ONLY_9917" not in json.dumps(aux.calls)
    assert len(snapshot(session)["operations"]) == 2
    assert not snapshot(session)["memories"]


def test_explicit_dedup_empty_search_and_schema_guard(make, tmp_path):
    session, _, _ = make()
    assert session.tool_registry.execute("recall", {"query": "cost center"}).status == "ok"
    for _ in range(2):
        result = session.tool_registry.execute("remember", {"fact": "Cost center is CC-8812."})
        assert result.status == "ok" and "Remembered successfully" in result.content
    assert len(snapshot(session)["memories"]) == 1
    session.close()
    with sqlite3.connect(tmp_path / "mem0" / "state.sqlite3") as db:
        db.execute("UPDATE settings SET value=? WHERE key='schema'", (json.dumps({**memory.SCHEMA, "dimensions": 12}),))
    second, _, _ = make()
    result = second.tool_registry.execute("recall", {"query": "cost center"})
    assert result.status == "tool_error" and "mismatch" in result.content


def test_tiny_budget_archives_without_oversized_aux_call(make):
    session, main, aux = make(context_budget=40)
    result = session.send("Employee number 40417.")
    assert result.status == "token_budget" and not main.calls and not aux.calls
    state = snapshot(session)
    assert state["operations"][0]["status"] == "fallback"


def test_no_memory_dir_stays_lazy(scripted, monkeypatch):
    def forbidden(*args):
        raise AssertionError("memory must remain uninitialized")
    monkeypatch.setattr(memory, "_new_backend", forbidden)
    session = adapters.run_agent_session(scripted(["Done."]), [], AgentConfig())
    assert session.send("Hello").status == "done" and session.memory is None
    session.close()


def test_reassertion_is_newer_without_duplicate_vector(make):
    session, _, _ = make(script=["Noted."] * 3)
    for fact in ["My cost center is CC-1111.", "My cost center is CC-8812.", "My cost center is CC-1111."]:
        session.send(fact)
    state = snapshot(session)
    assert len(state["memories"]) == 2
    result = session.tool_registry.execute("recall", {"query": "my cost center"})
    assert result.content.index("CC-1111") < result.content.index("CC-8812")
    assert "last_observed_at" in result.content


def test_aux_none_routes_extraction_to_main_callable(scripted, tmp_path):
    aux = Extractor()
    main = scripted(["Noted."])
    def shared(messages):
        if messages[0]["content"].startswith("Extract durable facts"):
            return aux(messages)
        return main(messages)
    session = adapters.run_agent_session(shared, [], AgentConfig(memory_dir=tmp_path))
    try:
        session.send("My cost center is CC-8812.")
        assert len(aux.calls) == 2 and len(main.calls) == 1
        assert "CC-8812" in snapshot(session)["memories"][0]["memory"]
    finally:
        session.close()


def test_same_directory_is_rejected_in_other_process(make, tmp_path):
    session, _, _ = make()
    session.send("My cost center is CC-8812.")
    program = """
import sys
from cs329z_hw1.memory import MemoryHandle, MemoryUnavailable
h = MemoryHandle(sys.argv[1])
try:
    with h.locked() as s:
        s._backend()
except MemoryUnavailable as exc:
    print(str(exc))
else:
    raise AssertionError('The directory must remain locked by its owner process')
finally:
    h.close()
"""
    child = subprocess.run([sys.executable, "-c", program, str(tmp_path)],
                           capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stderr
    assert "already accessed" in child.stdout
    assert session.tool_registry.execute("recall", {"query": "cost center"}).status == "ok"
