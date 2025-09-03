# rating.py
from __future__ import annotations
import numpy as np
import pandas as pd

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def simple_inverse(risk_prob: pd.Series | np.ndarray) -> pd.Series:
    rp = np.asarray(risk_prob, dtype=float)
    score = 10.0 * (1.0 - clip01(rp))
    return pd.Series(score, index=getattr(risk_prob, "index", None))

def percentile_inverse(risk_prob: pd.Series, reference: pd.Series) -> pd.Series:
    ref = pd.Series(reference, dtype=float).dropna()
    if ref.empty:
        return simple_inverse(risk_prob)
    ranks = risk_prob.apply(lambda v: (ref <= v).mean())
    score = 10.0 * (1.0 - ranks)
    return score

def score_to_band(score: float) -> str:
    if score >= 9.0:
        return "Prime (≈ AAA/AA)"
    if score >= 8.0:
        return "Strong (≈ A)"
    if score >= 7.0:
        return "Upper-Medium (≈ BBB)"
    if score >= 6.0:
        return "Speculative (≈ BB)"
    if score >= 4.5:
        return "Highly Speculative (≈ B)"
    if score >= 3.0:
        return "Substantial Risk (≈ CCC)"
    if score >= 2.0:
        return "Extremely Speculative (≈ CC/C)"
    return "Default-like (≈ D)"
