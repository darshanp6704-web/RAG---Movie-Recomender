"""Standalone demo showing how FastEmbed converts movie plots into vectors and ranks by cosine distance."""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from src import config

PLOTS: list[str] = [
    "A devoted dog waits at a train station every day for years after his owner passes away.",
    "Two estranged siblings reunite at their dying mother's bedside to make peace with the past.",
    "A terminally ill teenager falls in love and writes letters to comfort her family after she is gone.",
    "A skilled thief enters people's dreams to plant an idea inside a corporate heir's subconscious.",
    "Three clumsy college roommates accidentally swap briefcases with a gang of bumbling jewel thieves.",
    "A lone astronaut stranded on Mars uses botany and engineering to survive until rescue arrives.",
]

QUERY: str = "I want something that makes me cry"


def main() -> None:
    """Embed sample movie plots and rank them by cosine distance to the query."""
    model = TextEmbedding(
        model_name=config.EMBEDDING_MODEL,
        cache_dir=str(config.FASTEMBED_CACHE_DIR),
    )

    plot_embeddings = np.array(list(model.embed(PLOTS)), dtype=np.float32)
    print(f"Embedding array shape: {plot_embeddings.shape}")
    print(f"First 8 numbers of plot[0] embedding: {np.round(plot_embeddings[0, :8], 6).tolist()}")

    query_embedding = np.array(next(model.embed([QUERY])), dtype=np.float32)

    # FastEmbed embeddings are L2-normalized, so cosine distance = 1 - dot(a, b).
    norms = np.linalg.norm(plot_embeddings, axis=1) * np.linalg.norm(query_embedding)
    cosine_similarity = (plot_embeddings @ query_embedding) / norms
    cosine_distance = 1.0 - cosine_similarity

    ranked = sorted(
        zip(cosine_distance, cosine_similarity, PLOTS),
        key=lambda item: item[0],
    )

    print(f"\nQuery: {QUERY!r}")
    print("Plots ranked by cosine distance (lowest = closest match):")
    for rank, (dist, sim, plot) in enumerate(ranked, start=1):
        print(f"  {rank}. distance={dist:.4f} (similarity={sim:.4f}) | {plot}")


if __name__ == "__main__":
    main()
