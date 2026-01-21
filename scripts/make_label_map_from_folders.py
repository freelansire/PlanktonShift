from pathlib import Path
import yaml

IFCB_DIR = Path("data/ifcb/images")
OUT = Path("src/mapping/label_map.yaml")

def sanitize(label: str) -> str:
    # Make a stable coarse label token (avoids spaces, weird chars)
    return (
        label.strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )

def main():
    if not IFCB_DIR.exists():
        raise SystemExit(f"Missing: {IFCB_DIR}")

    raw_classes = sorted([p.name for p in IFCB_DIR.iterdir() if p.is_dir()])
    if not raw_classes:
        raise SystemExit("No class folders found under data/ifcb/images")

    # Identity mapping: raw -> sanitized (so YAML is clean + consistent)
    ifcb_map = {raw: sanitize(raw) for raw in raw_classes}

    # Placeholder for zooscan (you’ll fill later when you add a second domain)
    zooscan_map = {}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"ifcb": ifcb_map, "zooscan": zooscan_map},
            f,
            sort_keys=True,
            allow_unicode=True,
        )

    print(f"Wrote: {OUT}")
    print(f"IFCB classes: {len(raw_classes)}")
    print("Example mappings:")
    for k in raw_classes[:10]:
        print(f"  {k} -> {ifcb_map[k]}")

if __name__ == "__main__":
    main()
