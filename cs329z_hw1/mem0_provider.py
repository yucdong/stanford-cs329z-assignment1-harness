"""mem0 provider loaded lazily, after telemetry/bootstrap are configured."""
from contextlib import contextmanager
from copy import deepcopy
import json

from cs329z_hw1.memory import EXTRACTION_INSTRUCTIONS, MemoryUnavailable, _configure_sdk, _emit
from cs329z_hw1.tokens import CHARS_PER_TOKEN, count_message_tokens

# Also protect direct imports of this provider (e.g. configuration diagnostics).
_configure_sdk()
from mem0.llms.base import LLMBase


class CallableLLM(LLMBase):
    @contextmanager
    def bind(self, aux, budget, messages, source, callback, store):
        self.aux, self.budget = aux, budget
        self.messages, self.source, self.callback, self.store = messages, source, callback, store
        self.extracted, self.failed = [], False
        try:
            yield self
        finally:
            self.aux = self.messages = self.callback = self.store = None

    def _requests(self):
        # Stock mem0 prompts include unbounded history. This ADD-only adapter
        # uses current operation data and deduplicates against the entire store.
        prompt = EXTRACTION_INSTRUCTIONS + f" Extract only facts attributed to {self.source}."
        retry = " Previous response was invalid. Return the specified JSON object only."
        capacity = self.budget - count_message_tokens([
            {"role": "system", "content": prompt + retry}, {"role": "user", "content": ""},
        ])
        if capacity < 32:
            raise MemoryUnavailable("context_budget cannot fit memory extraction instructions.")
        payload = json.dumps(self.messages, ensure_ascii=False)
        if len(payload) <= capacity * CHARS_PER_TOKEN:
            return prompt, retry, [payload]
        pieces = []
        # Retain tails and roles; overlap supplies context across boundaries.
        # Reserve worst-case JSON escaping, measured with the project counter.
        for message in self.messages:
            overhead = len(json.dumps({"role": message["role"], "content": ""}, ensure_ascii=False))
            chars = max(1, (capacity * CHARS_PER_TOKEN - overhead) // 6)
            for start in range(0, len(message["content"]), max(1, chars - min(80, chars // 5))):
                pieces.append(json.dumps({"role": message["role"],
                    "content": message["content"][start:start + chars]}, ensure_ascii=False))
        return prompt, retry, pieces

    def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
        if not getattr(self, "aux", None):
            raise MemoryUnavailable("No session auxiliary model is bound to this memory operation.")
        try:
            prompt, retry, pieces = self._requests()
            facts = []
            for piece in pieces:
                for attempt in range(2):
                    request = [{"role": "system", "content": prompt + (retry if attempt else "")},
                               {"role": "user", "content": piece}]
                    if count_message_tokens(request) > self.budget:
                        raise MemoryUnavailable("Memory extraction request exceeds context_budget.")
                    response = self.aux(request)
                    _emit(self.callback, "extract", source=self.source, attempt=attempt + 1,
                          messages=deepcopy(request), response=response)
                    try:
                        data = json.loads(response)
                        if not isinstance(data, dict) or not isinstance(data.get("memory"), list):
                            raise ValueError("Expected a memory array.")
                        for fact in data["memory"]:
                            if (not isinstance(fact, dict) or not isinstance(fact.get("id"), str)
                                or not isinstance(fact.get("text"), str) or not fact["text"].strip()
                                or fact.get("attributed_to") not in ("user", "assistant")):
                                raise ValueError("Invalid memory item schema.")
                        facts.extend(f for f in data["memory"] if f["attributed_to"] == self.source)
                        break
                    except (ValueError, TypeError) as exc:
                        _emit(self.callback, "extract_validation", "invalid", source=self.source, error=str(exc))
                        if attempt:
                            raise MemoryUnavailable("Invalid extraction JSON after one retry.") from exc
            unique = {}
            for fact in facts:
                for chunk in self.store.chunks(fact["text"]):
                    unique[chunk] = {**fact, "text": chunk}
            self.extracted = list(unique.values())
            new = [f for f in self.extracted if not self.store._existing(f["text"])]
            _emit(self.callback, "extracted", source=self.source, results=self.extracted)
            return json.dumps({"memory": new}, ensure_ascii=False)
        except Exception as exc:
            self.failed = True
            _emit(self.callback, "extract", "unavailable", source=self.source,
                  error=f"{type(exc).__name__}: {exc}")
            raise
