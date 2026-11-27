"""Phase 3 embedding generation (FastEmbed ONNX) and persistent ChromaDB indexing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

from src import config

DEFAULT_BATCH_SIZE: int = 128


def get_embedder(cache_dir: Path | None = None) -> TextEmbedding:
    """Initialize FastEmbed ONNX model BAAI/bge-small-en-v1.5."""
    target_cache = cache_dir or config.FASTEMBED_CACHE_DIR
    target_cache.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(
        model_name=config.EMBEDDING_MODEL,
        cache_dir=str(target_cache),
    )


def get_chroma_client(chroma_dir: Path | None = None) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client rooted at chroma_dir."""
    target_dir = chroma_dir or config.CHROMA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(target_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str = config.CHROMA_COLLECTION,
) -> chromadb.Collection:
    """Get or create the cosine-distance ChromaDB collection."""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents(jsonl_path: Path | None = None) -> list[dict[str, Any]]:
    """Load prepared Phase 2 documents from JSONL."""
    src_path = jsonl_path or config.PROCESSED_DOCUMENTS_JSONL_PATH
    if not src_path.exists():
        raise FileNotFoundError(
            f"Prepared documents not found at '{src_path}'. "
            "Run Phase 2 (`python -m src.documents`) first."
        )
    with src_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_vector_index(
    jsonl_path: Path | None = None,
    chroma_dir: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedder: TextEmbedding | None = None,
) -> dict[str, Any]:
    """Embed all documents with FastEmbed and idempotently upsert into ChromaDB."""
    docs = load_documents(jsonl_path)
    model = embedder or get_embedder()
    client = get_chroma_client(chroma_dir)
    collection = get_or_create_collection(client)

    start = time.perf_counter()
    total = len(docs)
    for idx in range(0, total, batch_size):
        batch = docs[idx : idx + batch_size]
        ids = [doc["id"] for doc in batch]
        texts = [doc["text"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        vectors = [vec.tolist() for vec in model.embed(texts, batch_size=batch_size)]
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    elapsed = round(time.perf_counter() - start, 2)
    return {
        "indexed_batch_rows": total,
        "collection_count": collection.count(),
        "elapsed_seconds": elapsed,
        "chroma_dir": str(chroma_dir or config.CHROMA_DIR),
    }


def run_qa_checks(
    chroma_dir: Path | None = None,
    embedder: TextEmbedding | None = None,
    sample_query: str = "feel good movie about friendship",
) -> dict[str, Any]:
    """Run Phase 3 QA: count, 384-d check, metadata filter, reload, and sample search."""
    target_dir = chroma_dir or config.CHROMA_DIR
    model = embedder or get_embedder()

    # 1. Reload fresh client (simulating process restart)
    reloaded_client = get_chroma_client(target_dir)
    collection = reloaded_client.get_collection(name=config.CHROMA_COLLECTION)
    count = collection.count()

    # 2. Verify embedding dimension == 384 and space == cosine
    sample_get = collection.get(limit=1, include=["embeddings", "metadatas", "documents"])
    embeddings = sample_get["embeddings"]
    dimension = len(embeddings[0]) if embeddings is not None and len(embeddings) > 0 else 0

    # 3. Metadata filter test
    filter_res = collection.get(
        where={
            "$and": [
                {"type": "Movie"},
                {"language": "Hindi"},
                {"free_to_watch": True},
            ]
        },
        limit=5,
        include=["metadatas"],
    )

    # 4. Sample semantic search
    query_vec = next(model.embed([sample_query])).tolist()
    search_t0 = time.perf_counter()
    search_res = collection.query(
        query_embeddings=[query_vec],
        n_results=5,
        where={"type": "Movie"},
        include=["metadatas", "distances", "documents"],
    )
    search_ms = round((time.perf_counter() - search_t0) * 1000, 2)

    hits = []
    for doc_id, meta, dist, doc_text in zip(
        search_res["ids"][0],
        search_res["metadatas"][0],
        search_res["distances"][0],
        search_res["documents"][0],
    ):
        hits.append(
            {
                "title_id": doc_id,
                "language": meta.get("language"),
                "release_year": meta.get("release_year"),
                "rating": meta.get("rating"),
                "distance": round(float(dist), 4),
                "snippet": doc_text[:140],
            }
        )

    return {
        "count": count,
        "dimension": dimension,
        "space": collection.metadata.get("hnsw:space") if collection.metadata else None,
        "metadata_filter_matches": len(filter_res["ids"]),
        "metadata_filter_sample_ids": filter_res["ids"],
        "reload_after_restart_count": count,
        "sample_query": sample_query,
        "sample_search_ms": search_ms,
        "sample_hits": hits,
    }


def main() -> None:
    """CLI entry point for building and verifying the ChromaDB vector index."""
    parser = argparse.ArgumentParser(description="Movie Buddy Phase 3 vector indexer")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=config.PROCESSED_DOCUMENTS_JSONL_PATH,
        help="Path to data/processed/documents.jsonl",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=config.CHROMA_DIR,
        help="Path to persistent ChromaDB directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for FastEmbed and ChromaDB upsert",
    )
    parser.add_argument(
        "--verify-idempotent",
        action="store_true",
        help="Run a second pass to verify idempotent upsert without duplication",
    )
    args = parser.parse_args()

    embedder = get_embedder()
    build1 = build_vector_index(
        jsonl_path=args.jsonl,
        chroma_dir=args.chroma_dir,
        batch_size=args.batch_size,
        embedder=embedder,
    )
    print(f"First build: {build1}")

    if args.verify_idempotent:
        build2 = build_vector_index(
            jsonl_path=args.jsonl,
            chroma_dir=args.chroma_dir,
            batch_size=args.batch_size,
            embedder=embedder,
        )
        print(f"Second (idempotent) build: {build2}")

    qa = run_qa_checks(chroma_dir=args.chroma_dir, embedder=embedder)
    print("\nPhase 3 QA Results:")
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
