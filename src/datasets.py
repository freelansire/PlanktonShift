# === file: src/datasets.py ===
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import numpy as np
import tensorflow as tf

from .utils import list_images_by_class

AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class DatasetSpec:
    name: str               # "ifcb" or "zooscan"
    images_dir: str         # e.g. "data/ifcb/images"
    label_map: Dict[str, str]  # raw -> coarse


def _decode_resize(path: tf.Tensor, image_size: int) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    return img


def _augment(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return tf.clip_by_value(img, 0.0, 1.0)


def build_index(spec: DatasetSpec) -> Tuple[List[str], List[str]]:
    """
    Build (paths, coarse_labels) from a folder structured as:
      spec.images_dir/<raw_class>/*.(png|jpg|jpeg|webp)

    Any raw class not in label_map -> "other"
    """
    pairs = list_images_by_class(spec.images_dir)
    paths: List[str] = []
    coarse: List[str] = []

    for p, raw in pairs:
        c = spec.label_map.get(raw, "other")
        paths.append(p)
        coarse.append(c)

    return paths, coarse


def make_label_vocab(*all_labels: List[str]) -> List[str]:
    uniq = sorted(set().union(*[set(x) for x in all_labels]))
    return uniq


def make_tf_dataset(
    paths: List[str],
    labels: List[str],
    vocab: List[str],
    image_size: int = 224,
    batch_size: int = 32,
    training: bool = True,
    shuffle_buffer: int = 4096,
) -> tf.data.Dataset:
    label_to_id = {lab: i for i, lab in enumerate(vocab)}
    y = np.array([label_to_id[l] for l in labels], dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(min(shuffle_buffer, len(paths)), reshuffle_each_iteration=True)

    def _map(p, yi):
        img = _decode_resize(p, image_size)
        if training:
            img = _augment(img)
        return img, yi

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def split_train_val(paths: List[str], labels: List[str], val_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = int(len(paths) * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    def take(ix):
        return [paths[i] for i in ix], [labels[i] for i in ix]

    return take(tr_idx), take(val_idx)


def filter_by_allowed(paths: List[str], labels: List[str], allowed_set: Set[str] | None):
    """
    Keep only samples whose label is in allowed_set.
    If allowed_set is None or empty, returns inputs unchanged.
    """
    if not allowed_set:
        return paths, labels

    keep_p, keep_l = [], []
    for p, l in zip(paths, labels):
        if l in allowed_set:
            keep_p.append(p)
            keep_l.append(l)
    return keep_p, keep_l
