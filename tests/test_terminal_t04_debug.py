"""Run only terminal t04 and persist every LM input/output under runs/."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import pytest

from cs329z_hw1 import adapters
from cs329z_hw1.llm import LM
from cs329z_hw1.tokens import count_message_tokens
from cs329z_hw1.types import AgentConfig
from tests.helpers import contains_any, load_fixture


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class TraceLM:
    def __init__(self, lm, name, directory):
        self.lm = lm
        self.name = name
        self.directory = directory
        self.records = []

    def __call__(self, messages):
        number = len(self.records) + 1
        prefix = f"{self.name}-{number:02d}"
        record = {
            "call": prefix,
            "input_tokens": count_message_tokens(messages),
            "input_file": f"{prefix}.input.json",
        }
        self.records.append(record)
        save_json(self.directory / record["input_file"], messages)
        print(f"\n[{prefix}] input_tokens={record['input_tokens']}", flush=True)
        started = perf_counter()
        try:
            response = self.lm(messages)
        except Exception as exc:
            record["error_file"] = f"{prefix}.error.json"
            save_json(self.directory / record["error_file"], {
                "type": type(exc).__name__, "message": str(exc),
                "partial_text": getattr(exc, "text", None),
            })
            raise
        else:
            record["output_file"] = f"{prefix}.output.json"
            save_json(self.directory / record["output_file"], response)
            print(response, flush=True)
            return response
        finally:
            record["seconds"] = round(perf_counter() - started, 3)
            record["response_model"] = self.lm.response_model


@pytest.mark.live
def test_t04_trace(archive):
    item = next(item for item in load_fixture("terminal_questions.json") if item["id"] == "t04")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S-%f+0800")
    directory = Path(__file__).resolve().parents[1] / "runs" / f"terminal-t04-{stamp}"
    directory.mkdir(parents=True)
    print(f"\nTrace directory: {directory}", flush=True)
    main = TraceLM(LM("grading", cache=False, tag="debug/t04"), "main", directory)
    aux = TraceLM(LM("grading", cache=False, tag="debug/t04/aux"), "aux", directory)
    config = AgentConfig(max_turns=15, aux_lm=aux)
    session = adapters.run_agent_session(main, None, config)
    report = {
        "case": "t04", "question": item["question"], "expected": item["answers"],
        "model": main.lm.model, "backend": main.lm.backend, "cache": False,
        "context_budget": config.context_budget, "correct": False,
    }
    try:
        result = session.send(item["question"])
        report.update(status=result.status, answer=result.text,
                      correct=contains_any(result.text, item["answers"]))
    except Exception as exc:
        report.update(status="exception", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["main_calls"] = main.records
        report["aux_calls"] = aux.records
        report["within_context_budget"] = all(
            r["input_tokens"] <= config.context_budget for r in main.records + aux.records
        )
        save_json(directory / "transcript.json", session.transcript)
        save_json(directory / "report.json", report)
        lines = [
            "# Terminal t04 debug report", "",
            f"Question: {item['question']}", "",
            f"Expected: {item['answers']}", "",
            f"Status: {report.get('status')}; correct: {report['correct']}; cache: disabled", "",
            f"Model: {main.lm.model}; backend: {main.lm.backend}", "",
            f"Within context budget: {report['within_context_budget']}", "",
            "## Final answer", "", report.get("answer", report.get("error", "")), "",
            "## Model calls", "",
        ]
        for r in main.records + aux.records:
            output = r.get("output_file", r.get("error_file"))
            lines.append(
                f"- {r['call']}: {r['input_tokens']} input tokens; "
                f"[input]({r['input_file']}) / [output or error]({output})"
            )
        lines += ["", "[Transcript](transcript.json) | [Structured report](report.json)", ""]
        (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"\nReport: {directory / 'report.md'}", flush=True)

    assert result.status == "done"
    assert report["correct"], f"Unexpected answer: {result.text}; report: {directory}"
    assert report["within_context_budget"], f"Context budget exceeded; report: {directory}"
