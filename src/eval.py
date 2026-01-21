# === file: src/eval.py ===
from __future__ import annotations
import argparse
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .datasets import DatasetSpec, build_index, make_label_vocab, make_tf_dataset, filter_by_allowed
from .utils import load_label_map, ensure_dir
from .metrics import softmax_np, ece_score, brier_score


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
    ap.add_argument("--out", default="reports/artifacts/eval.json")
    args = ap.parse_args()

    allowed = set([x.strip() for x in args.allowed.split(",") if x.strip()])

    lm = load_label_map(args.label_map)
    spec = DatasetSpec(args.dataset, f"data/{args.dataset}/images", lm.get(args.dataset, {}))

    paths, labels = build_index(spec)
    paths, labels = filter_by_allowed(paths, labels, allowed)

    if len(paths) == 0:
        raise RuntimeError("No samples after filtering. Check label_map.yaml and --allowed.")

    # Build a consistent vocab for this dataset's filtered label space
    vocab = make_label_vocab(labels)
    label_to_id = {lab: i for i, lab in enumerate(vocab)}
    all_label_ids = list(range(len(vocab)))

    ds = make_tf_dataset(paths, labels, vocab, args.image_size, args.batch_size, training=False)

    model = tf.keras.models.load_model(args.model)
    logits_list = []
    y_list = []

    for xb, yb in ds:
        lg = model(xb, training=False).numpy()
        logits_list.append(lg)
        y_list.append(yb.numpy())

    logits = np.concatenate(logits_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)

    probs = softmax_np(logits)
    y_pred = probs.argmax(axis=1)

    # IMPORTANT: pass explicit labels so sklearn won't mismatch target_names
    rep = classification_report(
        y_true,
        y_pred,
        labels=all_label_ids,
        target_names=vocab,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=all_label_ids).tolist()

    out = {
        "dataset": args.dataset,
        "n": int(len(y_true)),
        "allowed": sorted(list(allowed)),
        "ece": ece_score(probs, y_true),
        "brier": brier_score(probs, y_true),
        "report": rep,
        "confusion_matrix": cm,
        "vocab": vocab,
    }

    ensure_dir("reports/artifacts")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", args.out)
    print("ECE:", out["ece"], "Brier:", out["brier"])


if __name__ == "__main__":
    main()
