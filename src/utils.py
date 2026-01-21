from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_label_map(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def list_images_by_class(root_dir: str) -> List[Tuple[str, str]]:
    """
    Returns: list of (filepath, raw_class_name)
    Expects: root_dir/<raw_class>/*.(png|jpg|jpeg)
    """
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    out: List[Tuple[str, str]] = []
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data folder not found: {root_dir}")

    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        raw = class_dir.name
        for fp in class_dir.rglob("*"):
            if fp.suffix.lower() in exts:
                out.append((str(fp), raw))
    if not out:
        raise RuntimeError(f"No images found under {root_dir}")
    return out
