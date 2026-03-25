import math
import numpy as np

def compute_confidence(reranked_docs):
    if not reranked_docs:
        return 0.0

    scores = [d['score'] for d in reranked_docs[:3]]

    import numpy as np

    # dominance-based confidence
    max_score = max(scores)
    mean_score = np.mean(scores)

    confidence = max_score - mean_score

    return float(max(0.0, min(confidence, 1.0)))