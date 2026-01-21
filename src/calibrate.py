# === file: src/calibrate.py ===
from __future__ import annotations
import argparse
import json
import numpy as np
import tensorflow as tf

from .datasets import DatasetSpec, build_index, split_train_val, make_label_vocab, make_tf_dataset, filter_by_allowed
from .utils import load_label_map, ensure_dir
from .metrics import softmax_np, ece_score, brier_score


def nll_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    probs = softmax_np(logits)
    p = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(p + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["ifcb", "zooscan"], required=True)
    ap.add_argument("--label_map", default="src/mapping/label_map.yaml")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--allowed",
        default="diatom,protist,detritus,artifact,other",
        help="Comma-separated coarse labels to keep (shared label space).",
    )
    ap.add_argument("--out", default="reports/artifacts/calibration.json")
    args = ap.parse_args()

    allowed = set([x.strip() for x in args.allowed.split(",") if x.strip()])

    lm = load_label_map(args.label_map)
    spec = DatasetSpec(args.dataset, f"data/{args.dataset}/images", lm.get(args.dataset, {}))

    paths, labels = build_index(spec)
    paths, labels = filter_by_allowed(paths, labels, allowed)

    if len(paths) < 50:
        raise RuntimeError("Too few samples after filtering for calibration. Add more images or relax --allowed.")

    vocab = make_label_vocab(labels)

    # Use a validation split for calibration
    (_, _), (va_p, va_y) = split_train_val(paths, labels, val_frac=0.2, seed=7)
    ds = make_tf_dataset(va_p, va_y, vocab, args.image_size, args.batch_size, training=False)

    model = tf.keras.models.load_model(args.model)

    logits_list = []
    y_list = []
    for xb, yb in ds:
        logits_list.append(model(xb, training=False).numpy())
        y_list.append(yb.numpy())

    logits = np.concatenate(logits_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Grid search temperature
    Ts = np.concatenate([np.linspace(0.5, 3.0, 26), np.linspace(3.0, 10.0, 15)])
    best = {"T": 1.0, "nll": 1e9}

    for T in Ts:
        scaled = logits / float(T)
        nll = nll_from_logits(scaled, y)
        if nll < best["nll"]:
            best = {"T": float(T), "nll": float(nll)}

    probs0 = softmax_np(logits)
    probsT = softmax_np(logits / best["T"])

    out = {
        "dataset": args.dataset,
        "allowed": sorted(list(allowed)),
        "T": best["T"],
        "val_n": int(len(y)),
        "nll_before": nll_from_logits(logits, y),
        "nll_after": best["nll"],
        "ece_before": ece_score(probs0, y),
        "ece_after": ece_score(probsT, y),
        "brier_before": brier_score(probs0, y),
        "brier_after": brier_score(probsT, y),
        "vocab": vocab,
    }

    ensure_dir("reports/artifacts")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", args.out)
    print(out)


if __name__ == "__main__":
    main()
