"""Replay course memory examples with fresh LM responses and export full traces.

Usage: uv run python scripts/trace_memory.py [all|the-spec|my-team|cost-center]
"""
import argparse
import copy
import dataclasses
import json
from pathlib import Path
import tempfile

from cs329z_hw1 import adapters
from cs329z_hw1.llm import LM
from tests.test_memory import PAIRS, run_pair
from tests.thresholds import MEMORY_MIN_CORRECT


def trace_pair(pair, root, use_cache=False):
    output = Path(tempfile.mkdtemp(prefix=pair["id"] + "-", dir=root)).resolve()
    trace = {"pair": pair, "cache_enabled": use_cache, "sessions": []}
    active = None

    def save():
        (output / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    class RecordingLM:
        def __init__(self, kind):
            self.kind = kind
            self.lm = LM(None, cache=use_cache, tag="memory-mem0/" + pair["id"] + "/" + kind)
            self.model = self.lm.model

        def __call__(self, messages):
            call = {"kind": self.kind, "messages": copy.deepcopy(messages)}
            active["calls"].append(call)
            try:
                call["response"] = self.lm(messages)
                print(f"{pair['id']} session {active['number']} {self.kind}: {call['response']}", flush=True)
                return call["response"]
            except Exception as exc:
                call["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                call["raw_response"] = copy.deepcopy(self.lm.last_response)
                save()

    original_send = adapters.AgentSession.send

    def traced_send(session, message):
        nonlocal active
        active = {"number": len(trace["sessions"]) + 1, "user_input": message, "calls": []}
        trace["sessions"].append(active)
        try:
            result = original_send(session, message)
            active["result"] = dataclasses.asdict(result)
            return result
        finally:
            active["transcript"] = copy.deepcopy(session.transcript)
            try:
                with session.memory.locked() as store:
                    active["memory_snapshot"] = store.snapshot()
            except Exception as exc:
                active["snapshot_error"] = str(exc)
            session.close()
            save()

    main, aux = RecordingLM("main"), RecordingLM("aux")
    trace.update(model=main.model, backend=main.lm.backend)
    adapters.AgentSession.send = traced_send
    try:
        trace["outcome"] = run_pair(pair, main, aux, output)
        return trace["outcome"]
    finally:
        adapters.AgentSession.send = original_send
        trace["usage"] = {"main": main.lm.usage(), "aux": aux.lm.usage()}
        save()
        lines = [f"# Memory trace: {pair['id']}", "", f"Model: {main.model}; cache: {use_cache}", ""]
        for session in trace["sessions"]:
            lines.extend([f"## Session {session['number']}", ""])
            for event in session["transcript"]:
                label = " ".join(str(event.get(k, "")) for k in ("type", "phase", "name", "status")).strip()
                content = event.get("content") or json.dumps(event, ensure_ascii=False, indent=2)
                lines.extend([f"### {label}", "", "````text", content, "````", ""])
            lines.extend(["### Persisted state", "", "````json",
                          json.dumps(session.get("memory_snapshot"), ensure_ascii=False, indent=2), "````", ""])
            for number, call in enumerate(session["calls"], 1):
                lines.extend([f"### Raw LM response: {call['kind']} call {number}", "",
                              "````json", json.dumps(call.get("raw_response"), ensure_ascii=False, indent=2),
                              "````", ""])
        lines.extend(["## Outcome", "", "````json", json.dumps(trace.get("outcome"), ensure_ascii=False, indent=2), "````"])
        (output / "trace.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Trace saved to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", nargs="?", default="all", choices=["all"] + [p["id"] for p in PAIRS])
    parser.add_argument("--use-cache", action="store_true", help="Opt in to LM cache; default is fresh calls.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "runs" / "memory-trace"
    root.mkdir(parents=True, exist_ok=True)
    rows = [trace_pair(p, root, args.use_cache) for p in PAIRS if args.pair in ("all", p["id"])]
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    correct = sum(r["ok"] for r in rows)
    minimum = MEMORY_MIN_CORRECT if args.pair == "all" else 1
    print(f"Memory live: {correct}/{len(rows)} passed; required: {minimum}.")
    return 0 if correct >= minimum else 1


if __name__ == "__main__":
    raise SystemExit(main())
