import os
from pathlib import Path
from datasets import load_dataset

OUT = Path("data/zooscan/images")
OUT.mkdir(parents=True, exist_ok=True)

# HF dataset mirror of ZooScanNet
ds = load_dataset("project-oceania/zooscannet", split="train")  # :contentReference[oaicite:6]{index=6}

# Try to detect common field names
label_col = "label" if "label" in ds.column_names else ("labels" if "labels" in ds.column_names else None)
image_col = "image" if "image" in ds.column_names else None
if label_col is None or image_col is None:
    raise RuntimeError(f"Unexpected columns: {ds.column_names}")

# If label is a ClassLabel feature, this gives names; otherwise fallback to str(label)
label_names = None
try:
    feat = ds.features[label_col]
    if hasattr(feat, "names"):
        label_names = feat.names
except Exception:
    pass

MAX_PER_CLASS = 5000  # adjust for a fast demo
counts = {}

for i, ex in enumerate(ds):
    lab = ex[label_col]
    lab_name = label_names[lab] if (label_names is not None and isinstance(lab, int) and lab < len(label_names)) else str(lab)

    counts.setdefault(lab_name, 0)
    if counts[lab_name] >= MAX_PER_CLASS:
        continue

    img = ex[image_col]  # PIL image
    class_dir = OUT / lab_name
    class_dir.mkdir(parents=True, exist_ok=True)

    fp = class_dir / f"{lab_name}_{counts[lab_name]:06d}.png"
    img.save(fp)
    counts[lab_name] += 1

print("Saved per-class counts (sampled):")
for k in sorted(counts, key=lambda x: counts[x], reverse=True)[:20]:
    print(k, counts[k])
