from __future__ import annotations
import numpy as np

def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def ece_score(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE).
    probs: (N, C) probabilities
    y_true: (N,) int labels
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += (m.mean()) * abs(acc[m].mean() - conf[m].mean())

    return float(ece)

def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    n, c = probs.shape
    y_onehot = np.zeros((n, c), dtype=np.float32)
    y_onehot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
