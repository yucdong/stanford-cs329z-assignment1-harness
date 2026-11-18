"""Document tool wiring and lazy shared index initialization."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import pytest

from cs329z_hw1 import adapters, cardinal, data
from cs329z_hw1.types import AgentConfig


@pytest.fixture
def load_docs(monkeypatch):
    monkeypatch.setattr(cardinal, "_doc_index", None)
    loader = Mock(return_value=[{
        "doc_id": "policy", "title": "Security Policy",
        "text": "Passwords require fourteen characters.\nUse a hardware key.",
    }])
    monkeypatch.setattr(data, "load_docs", loader)
    return loader


def test_registered_document_tools_share_lazy_index(load_docs):
    session = adapters.run_agent_session(lambda messages: "unused", None, AgentConfig())
    load_docs.assert_not_called()
    search = session.tool_registry.execute("search_docs", {"query": "passwords"})
    assert search.status == "ok"
    assert "policy" in search.content
    assert "fourteen" in search.content
    result = session.tool_registry.execute("read_doc", {"doc_id": "policy", "start_line": 2})
    assert result.status == "ok"
    assert "Use a hardware key." in result.content
    load_docs.assert_called_once_with()


def test_concurrent_initialization_builds_once(load_docs, monkeypatch):
    build = Mock(wraps=adapters.run_build_doc_index)
    monkeypatch.setattr(adapters, "run_build_doc_index", build)
    barrier = Barrier(4)

    def get_index(_):
        barrier.wait(timeout=5)
        return cardinal.get_doc_index()

    with ThreadPoolExecutor(max_workers=4) as pool:
        indexes = list(pool.map(get_index, range(4)))
    assert all(index is indexes[0] for index in indexes)
    build.assert_called_once()
    load_docs.assert_called_once_with()


def test_failed_initialization_can_retry(load_docs, monkeypatch):
    build = Mock(side_effect=[RuntimeError("temporary failure"), object()])
    monkeypatch.setattr(adapters, "run_build_doc_index", build)
    with pytest.raises(RuntimeError, match="temporary failure"):
        cardinal.get_doc_index()
    assert cardinal._doc_index is None
    index = cardinal.get_doc_index()
    assert cardinal.get_doc_index() is index
    assert build.call_count == 2
