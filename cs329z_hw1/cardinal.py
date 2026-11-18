"""Course-provided definition of the Cardinal toolset.

``CARDINAL_TOOLS`` fixes, for each of the ten tools your agent offers, its
name, the JSON Schema of its arguments, and whether it is read-only. The
tests and the simulated users rely on these names and argument names, so do
not change them. You may register additional tools of your own.

Two things are left to you: the implementation of each tool, and its
description (the text the model reads to decide when and how to call it).
``cardinal_spec`` combines the two with the fixed parts into a ``ToolSpec``:

    spec = cardinal_spec("search_docs", "Search the ... (your text)", my_search_docs)
"""

from __future__ import annotations

import copy
from threading import Lock
from typing import Any, Callable

from cs329z_hw1.types import ToolSpec


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_STRING = {"type": "string"}
_K = {"type": "integer", "minimum": 1, "default": 5}

# name -> {"parameters": JSON Schema, "read_only": bool}, in a fixed order.
# A tool that is not read-only needs the user's approval in confirm mode.
# ``remember`` writes only to the agent's own memory store, so it is marked
# read-only and does not need approval.
CARDINAL_TOOLS: dict[str, dict[str, Any]] = {
    "search_emails": {
        "parameters": _schema({"query": _STRING, "k": _K}, ["query"]),
        "read_only": True,
    },
    "read_email": {
        "parameters": _schema({"email_id": _STRING}, ["email_id"]),
        "read_only": True,
    },
    "email_qa": {
        "parameters": _schema({"question": _STRING}, ["question"]),
        "read_only": True,
    },
    "daily_digest": {
        "parameters": _schema(
            {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}}, ["date"]
        ),
        "read_only": True,
    },
    "search_docs": {
        "parameters": _schema({"query": _STRING, "k": _K}, ["query"]),
        "read_only": True,
    },
    "read_doc": {
        "parameters": _schema(
            {"doc_id": _STRING, "start_line": {"type": "integer", "minimum": 1, "default": 1}},
            ["doc_id"],
        ),
        "read_only": True,
    },
    "run_terminal": {
        "parameters": _schema({"cmd": _STRING}, ["cmd"]),
        "read_only": False,
    },
    "remember": {
        "parameters": _schema({"fact": _STRING}, ["fact"]),
        "read_only": True,
    },
    "recall": {
        "parameters": _schema({"query": _STRING}, ["query"]),
        "read_only": True,
    },
    "ask_user": {
        "parameters": _schema({"question": _STRING}, ["question"]),
        "read_only": True,
    },
}


def cardinal_spec(name: str, description: str, fn: Callable[..., Any]) -> ToolSpec:
    """Build the ``ToolSpec`` for the Cardinal tool called ``name``.

    ``description`` and ``fn`` are yours. The parameter schema and the
    read-only flag come from ``CARDINAL_TOOLS``. ``fn`` is called with the
    arguments as keyword arguments, for example ``fn(query="...", k=3)``.
    Optional arguments the model leaves out are not filled in, so give them
    default values in ``fn``'s signature (k=5, start_line=1).
    """
    if name not in CARDINAL_TOOLS:
        raise KeyError(f"{name!r} is not a Cardinal tool. Known tools: {', '.join(CARDINAL_TOOLS)}")
    entry = CARDINAL_TOOLS[name]
    return ToolSpec(
        name=name,
        description=description,
        parameters=copy.deepcopy(entry["parameters"]),
        fn=fn,
        read_only=entry["read_only"],
    )

_doc_index = None
_doc_index_lock = Lock()


def get_doc_index():
    """Load and build the shared document index on the first tool call."""
    global _doc_index

    with _doc_index_lock:
        if _doc_index is None:
            # adapters imports cardinal's specs, so defer the reverse import
            # until both modules have finished initializing.
            from cs329z_hw1.adapters import run_build_doc_index
            from cs329z_hw1.data import load_docs

            _doc_index = run_build_doc_index(load_docs())
        return _doc_index


def build_index_if_needed() -> None:
    """Initialize the document index without rebuilding an existing one."""
    get_doc_index()

def search_emails(query: str, k: int = 5) -> Any:
    """Search emails with the given query and return up to k results."""
    pass

def read_email(email_id: str) -> Any:
    """Read the email with the given ID."""
    pass

def email_qa(question: str) -> Any:
    """Answer a question about emails."""
    pass

def daily_digest(date: str) -> Any:
    """Get the daily digest for the given date."""
    pass

def search_docs(query: str, k: int = 5) -> Any:
    """Search documents with the given query and return up to k results."""
    from cs329z_hw1.adapters import run_search_docs

    return run_search_docs(get_doc_index(), query, k)

def read_doc(doc_id: str, start_line: int = 1) -> Any:
    """Read the document with the given ID starting from the specified line."""
    from cs329z_hw1.adapters import run_read_doc

    return run_read_doc(get_doc_index(), doc_id, start_line)

def run_terminal(cmd: str) -> Any:
    """Run the given terminal command."""
    pass

def remember(fact: str) -> Any:
    """Remember the given fact."""
    pass

def recall(query: str) -> Any:
    """Recall information based on the given query."""
    pass

def ask_user(question: str) -> Any:
    """Ask the user the given question."""
    pass


MEMORY_DIR_SPECS = [
    cardinal_spec("remember", "Remember the given fact.", remember),
    cardinal_spec("recall", "Recall information based on the given query.", recall),
]

USER_SPECS = [
    cardinal_spec("ask_user", "Ask the user the given question.", ask_user),
]

CARDINAL_SPECS = [
    cardinal_spec("search_emails", "Search emails with the given query and return up to k results.", search_emails),
    cardinal_spec("read_email", "Read the email with the given ID.", read_email),
    cardinal_spec("email_qa", "Answer a question about emails.", email_qa),
    cardinal_spec("daily_digest", "Get the daily digest for the given date.", daily_digest),
    cardinal_spec("search_docs", "Search documents with the given query and return up to k results.", search_docs),
    cardinal_spec("read_doc", "Read the document with the given ID starting from the specified line.", read_doc),
    cardinal_spec("run_terminal", "Run the given terminal command.", run_terminal),
    *MEMORY_DIR_SPECS,
    *USER_SPECS,
]
