# Local automatic memory

Install with `uv sync`. The lockfile pins `mem0ai==2.2.1` and
`fastembed==0.9.1`. It was resolved through the Tsinghua PyPI mirror because
`files.pythonhosted.org` failed TLS negotiation in the development environment.

```python
from pathlib import Path
from cs329z_hw1.adapters import run_agent_session
from cs329z_hw1.types import AgentConfig

session = run_agent_session(main_lm, None, AgentConfig(
    memory_dir=Path("my-memory"), aux_lm=aux_lm,
))
try:
    session.send("My cost center is CC-8812.")
finally:
    session.close()
```

Every `send` retrieves previous history before indexing the new input, injects
up to five results within 1,024 project-counted tokens and the available context
budget, archives the original user input, and extracts user facts. Normal final
answers are archived and separately processed for new factual conclusions.
Current memory context is retained during conversation compaction. Tool feedback,
errors and budget-stop messages are never submitted as new user messages.
`remember(fact)` and `recall(query)` remain available; their results are separate
from automatic `memory` transcript events.

Each canonical `memory_dir` contains:

| Path | Contents |
| --- | --- |
| `mem0/qdrant/` | Local vectors, fact payloads and any entity collection |
| `mem0/history.sqlite3` | mem0 message and fact history |
| `mem0/state.sqlite3` | Full originals, operation status, fingerprints and index schema |

The collection is `knowledge`, scoped to `local-user`. Session IDs are provenance
metadata, never retrieval filters. An incompatible schema/model/dimension is an
error. Directory leases share one locked backend in the current process; closing
the last session releases both SQLite and the Qdrant client. Concurrent processes
cannot open the same Qdrant directory. No remote database fallback is configured.

The local model is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions,
128-token model limit). First use downloads about 220 MB of public weights;
FastEmbed's weight cache can be shared. mem0 may also cache its local BM25 encoder.
Telemetry is disabled before import. Import-time SDK metadata is placed in
`~/.cache/cs329z/mem0-sdk` unless `MEM0_DIR` was already set; this location is never
used for knowledge or switched between directories. The optional spaCy extras are
not installed: mem0 falls back to original text for lexical matching and skips
spaCy entity extraction. Dense multilingual retrieval remains enabled.

The custom LLM provider calls the **current session's** `aux_lm` (or `main_lm` when
it is absent). It creates no inference API client. Because mem0's stock prompt
contains unbounded history, the provider uses compact extraction instructions and
current-operation messages, splitting oversized requests within `context_budget`.
Responses must contain a `memory` array of `{id, text, attributed_to}` objects;
invalid JSON/schema is retried once. Deduplication checks the entire local store,
including writes committed before an interrupted journal update. Extraction in
this mem0 version adds facts; old facts are not automatically deleted. Reassertions
record observation time without duplicate vectors. Answers should prefer explicit
newer user corrections; retrieval is not a guaranteed conflict-resolution system.

If extraction fails, user originals are indexed as labeled raw fallback data.
Long raw data uses tokenizer-offset windows of at most 120 tokens with 20-token
overlap; additional sentence units prevent short facts from being diluted by
unrelated text. Assistant extraction failures preserve the reply in the journal
without indexing the entire reply as truth. Backend failures leave durable pending
operations for recovery on later access. Main tasks continue; explicit tool
failures return `tool_error`, and unavailable memory is distinct from an empty
search. Raw archival cannot be guaranteed when the directory itself is unwritable.

Existing `memory.json["documents"]` is imported verbatim without LLM inference;
the original file is preserved. Stable migration operation IDs and vector checks
make interrupted retries idempotent. No old BM25 index is imported.

Run the deterministic integration tests (real local embeddings, scripted LMs):

```powershell
uv run pytest tests/test_memory.py tests/test_mem0_integration.py tests/test_agent_loop.py tests/test_compaction.py tests/test_compaction_boundaries.py -m "not live"
```

Replay all three course live pairs, using the existing answer assertions:

```powershell
uv run python scripts/trace_memory.py all
```

LM caching is disabled by default; `--use-cache` explicitly enables it. Each pair
exports `trace.md` and `trace.json` under `runs/memory-trace/`, including automatic
retrieval, injected context, auxiliary requests/responses, persistence events,
explicit tool calls, final responses and database snapshots. Each session closes
before the next starts. The aggregate exit status uses the course's
`MEMORY_MIN_CORRECT` threshold (currently two of three); each case's individual
pass/fail remains visible. Course grading environments that copy only `cs329z_hw1/`
must also install the pinned dependencies and provide/download the local weights.

Uncached LM calls also save raw SDK responses and all received streaming events
in `raw_responses.jsonl` beside the LM cost ledger (by default
`.lm_cache/copilot-bridge/` for the bridge). A `call_id` links the two logs.
The memory exporter includes these records in both trace formats, including
failed calls, so streamed text can be compared with the terminal response and
the text delivered to the agent. Cache hits have no fresh raw response.

The main agent retries consecutive empty or whitespace-only responses twice,
using the same input; every attempt counts against `max_turns` and `token_budget`.
Retries appear as `model_retry` transcript events. Three consecutive empty
responses stop with `malformed`, never `done`, and are not saved as assistant
facts. A nonempty response or a new `send` resets the empty-response counter.
Auxiliary extraction and compaction retain their existing retry rules.
