from typing import List

def compute_confidence(reranked_docs: List[dict]) -> None:
    """
    Compute retrieval confidence using reranker scores
    """

    if not reranked_docs:
        return 0.0
    
    # take top K score
    top_k = reranked_docs[:3]
    scores = [d['score'] for d in top_k]

    # Normalize scores (cross-encoder usually ~[-5, +5] or [0,1])
    # We'll use sigmoid normalization
    import math
    norm_scores = [1 / (1 + math.exp(-s)) for s in scores]

    # Final confidence = average
    return sum(norm_scores) / len(norm_scores)