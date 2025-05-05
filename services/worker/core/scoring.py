import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def score_embedding(emb, profile, tier):
    if tier == "basic":
        score = float(cosine_similarity([emb], [profile.mean()])[0,0])
        match = score > 0.65
    # medium / high: cohort-norm, PLDA…
    return score, match