"""Free-text recall on the default backend.

mem0 is opt-in, so most installs use the SQLite path. It used to match the whole
query as one LIKE pattern — "why was a sample excluded?" found nothing even when
a note said exactly that in other words — which made the default backend a
fallback in name only.
"""

from __future__ import annotations

import pytest

from skinmcp.config import CONFIG
from skinmcp.tools import memory_tools


@pytest.fixture()
def notes(project):
    old = CONFIG.memory_backend
    CONFIG.memory_backend = "sqlite"
    for tag, body in (
        ("qc", "Dropped sample S7 for high ambient RNA"),
        ("clustering", "Used resolution 0.4 after 0.8 over-split the neutrophils"),
        ("de", "Pseudobulk not possible at D19: only two Sham samples"),
    ):
        memory_tools.note(tag=tag, body=body, project_id=project)
    yield project
    CONFIG.memory_backend = old


def _hits(project, q):
    r = memory_tools.search(query=q, project_id=project)
    assert r["ok"], r.get("error")
    return r["summary"].get("hits", [])


def test_a_natural_question_finds_the_note(notes):
    hits = _hits(notes, "why was a sample excluded?")
    assert hits, "no hit for a question that shares 'sample' with a note"
    assert "S7" in hits[0]["snip"]


def test_results_are_ranked_by_words_matched(notes):
    hits = _hits(notes, "what resolution did we use for clustering?")
    assert hits[0]["title"] == "clustering"
    assert hits[0]["score"] == 1.0


def test_stopwords_alone_do_not_match_everything(notes):
    """"the was and" carries no signal; matching on it would rank all rows equal."""
    assert _hits(notes, "the was and of") == []


def test_an_unrelated_query_finds_nothing(notes):
    assert _hits(notes, "spatial transcriptomics deconvolution") == []


def test_backend_is_reported(notes):
    hits = _hits(notes, "ambient RNA")
    assert hits and hits[0]["backend"] == "keyword"
