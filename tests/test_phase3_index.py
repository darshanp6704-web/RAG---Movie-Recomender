"""Phase 3 tests for FastEmbed ONNX embeddings, ChromaDB persistence, and QA."""

from __future__ import annotations

import chromadb

from src import config
from src.index import (
    get_chroma_client,
    get_embedder,
    get_or_create_collection,
    run_qa_checks,
)


def test_chroma_collection_exists_with_cosine_and_3000_docs() -> None:
    """Verify persistent ChromaDB collection has 3000 docs, 384-d embeddings, and cosine space."""
    assert config.CHROMA_DIR.exists(), "Run `python -m src.index` first."
    client = get_chroma_client(config.CHROMA_DIR)
    col = client.get_collection(name=config.CHROMA_COLLECTION)

    assert col.count() == 3000
    assert col.metadata is not None
    assert col.metadata.get("hnsw:space") == "cosine"

    sample = col.get(ids=["m_1726"], include=["embeddings", "metadatas", "documents"])
    assert len(sample["ids"]) == 1
    assert len(sample["embeddings"][0]) == 384
    assert sample["metadatas"][0]["language"] == "English"
    assert sample["metadatas"][0]["type"] == "Movie"
    assert "Iron Man" in sample["documents"][0]


def test_idempotent_upsert_does_not_duplicate(tmp_path) -> None:
    """Verify re-upserting the same IDs does not create duplicates."""
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = get_or_create_collection(client, "test_idempotent")
    emb = [0.1] * 384
    for _ in range(2):
        col.upsert(
            ids=["m_1"],
            embeddings=[emb],
            documents=["Test friendship movie"],
            metadatas=[{"title_id": "m_1", "type": "Movie"}],
        )
    assert col.count() == 1


def test_qa_filter_reload_and_friendship_search() -> None:
    """Verify metadata filtering, fresh client reload, and sample search QA."""
    embedder = get_embedder()
    qa = run_qa_checks(
        chroma_dir=config.CHROMA_DIR,
        embedder=embedder,
        sample_query="feel good movie about friendship",
    )
    assert qa["count"] == 3000
    assert qa["reload_after_restart_count"] == 3000
    assert qa["dimension"] == 384
    assert qa["space"] == "cosine"
    assert qa["metadata_filter_matches"] > 0
    assert len(qa["sample_hits"]) == 5
    # Verify cosine distances are well-formed and sorted ascending
    distances = [hit["distance"] for hit in qa["sample_hits"]]
    assert distances == sorted(distances)
