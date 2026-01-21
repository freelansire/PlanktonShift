# === file: src/train.py ===
from __future__ import annotations

import argparse
import os
import numpy as np
import tensorflow as tf

from .datasets import (
    DatasetSpec,
    build_index,
    make_label_vocab,
    split_train_val,
    make_tf_dataset,
    filter_by_allowed,
)
from .model import build_model, set_backbone_trainable
from .utils import load_label_map, ensure_dir


def coral_loss(source_feat: tf.Tensor, target_feat: tf.Tensor) -> tf.Tensor:
    """CORAL: align covariance of embeddings."""
    def cov(x):
        x = x - tf.reduce_mean(x, axis=0, keepdims=True)
        n = tf.cast(tf.shape(x)[0], tf.float32)
        return tf.matmul(x, x, transpose_a=True) / (n - 1.0 + 1e-6)

    cs = cov(source_feat)
    ct = cov(target_feat)
    d = tf.cast(tf.shape(cs)[0], tf.float32)
    return tf.reduce_sum(tf.square(cs - ct)) / (4.0 * d * d + 1e-6)


def make_domain_iter(ds: tf.data.Dataset):
    return iter(ds.repeat())


def normalize_outpath(out: str) -> str:
    """
    Keras 3: model.save requires .keras or .h5.
    We standardize on .keras and allow user to pass a path without extension.
    """
    if out.endswith(".keras") or out.endswith(".h5"):
        return out
    return out + ".keras"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ifcb", "zooscan"], required=True)
    ap.add_argument("--target", choices=["ifcb", "zooscan"], required=True)
    ap.add_argument("--label_map", default="src/mapping/label_map.yaml")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--use_coral", action="store_true")
    ap.add_argument("--coral_weight", type=float, default=0.2)

    ap.add_argument(
        "--allowed",
        default="diatom,protist,detritus,artifact,other",
        help="Comma-separated coarse labels to keep (shared label space).",
    )

    ap.add_argument("--out", default="models/planktonshift_source")
    args = ap.parse_args()

    out_path = normalize_outpath(args.out)
    out_dir = os.path.dirname(out_path) or "models"
    ensure_dir(out_dir)

    allowed = set([x.strip() for x in args.allowed.split(",") if x.strip()])

    lm = load_label_map(args.label_map)
    source_spec = DatasetSpec(args.source, f"data/{args.source}/images", lm.get(args.source, {}))
    target_spec = DatasetSpec(args.target, f"data/{args.target}/images", lm.get(args.target, {}))

    s_paths, s_labels = build_index(source_spec)
    t_paths, t_labels = build_index(target_spec)

    # Filter both domains to the shared coarse label space
    s_paths, s_labels = filter_by_allowed(s_paths, s_labels, allowed)
    t_paths, t_labels = filter_by_allowed(t_paths, t_labels, allowed)

    print(f"After filtering: source={len(s_paths)} target={len(t_paths)} allowed={sorted(list(allowed))}")

    if len(s_paths) < 100:
        raise RuntimeError("Too few source samples after filtering. Check label_map.yaml and --allowed.")

    # Shared vocabulary
    vocab = make_label_vocab(s_labels, t_labels) if len(t_labels) > 0 else make_label_vocab(s_labels)
    print("Coarse vocab:", vocab)

    (s_tr_p, s_tr_y), (s_va_p, s_va_y) = split_train_val(s_paths, s_labels, val_frac=0.15, seed=42)
    s_tr = make_tf_dataset(s_tr_p, s_tr_y, vocab, args.image_size, args.batch_size, training=True)
    s_va = make_tf_dataset(s_va_p, s_va_y, vocab, args.image_size, args.batch_size, training=False)

    # Target stream for CORAL (labels ignored)
    if args.use_coral:
        if len(t_paths) < args.batch_size:
            raise RuntimeError("Too few target samples for CORAL. Add more target images or reduce batch_size.")
        t_stream = make_tf_dataset(t_paths, t_labels, vocab, args.image_size, args.batch_size, training=True)
        t_iter = make_domain_iter(t_stream)
    else:
        t_iter = None

    model, feature_model = build_model(num_classes=len(vocab), image_size=args.image_size)

    # Stage 1: train head only
    set_backbone_trainable(model, trainable=False)

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    tr_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    va_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(x_s, y_s):
        with tf.GradientTape() as tape:
            logits = model(x_s, training=True)
            loss = ce(y_s, logits)

            if args.use_coral:
                x_t, _ = next(t_iter)  # type: ignore[misc]
                fs = feature_model(x_s, training=True)
                ft = feature_model(x_t, training=True)
                loss = loss + args.coral_weight * coral_loss(fs, ft)

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        tr_acc.update_state(y_s, logits)
        return loss

    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        loss = ce(y, logits)
        va_acc.update_state(y, logits)
        return loss

    best_va = 0.0
    for ep in range(1, args.epochs + 1):
        tr_acc.reset_state()
        va_acc.reset_state()

        tr_losses = []
        for xb, yb in s_tr:
            l = train_step(xb, yb)
            tr_losses.append(float(l.numpy()))

        va_losses = []
        for xb, yb in s_va:
            l = val_step(xb, yb)
            va_losses.append(float(l.numpy()))

        print(
            f"Epoch {ep:02d} | tr_loss={np.mean(tr_losses):.4f} tr_acc={tr_acc.result().numpy():.4f} "
            f"| va_loss={np.mean(va_losses):.4f} va_acc={va_acc.result().numpy():.4f}"
        )

        if va_acc.result().numpy() > best_va:
            best_va = float(va_acc.result().numpy())
            model.save(out_path)  # Keras 3 safe
            print("Saved best model to", out_path)

    # Stage 2: light fine-tuning
    print("Fine-tuning backbone...")
    model = tf.keras.models.load_model(out_path)
    set_backbone_trainable(model, trainable=True, fine_tune_at=60)
    opt2 = tf.keras.optimizers.Adam(learning_rate=args.lr * 0.2)

    model.compile(
        optimizer=opt2,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.fit(s_tr, validation_data=s_va, epochs=3, verbose=1)
    model.save(out_path)
    print("Final saved to", out_path)


if __name__ == "__main__":
    main()
