from pathlib import Path
import yaml

ZOOSCAN_DIR = Path("data/zooscan/images")  # this is your PlanktonSet folder
OUT = Path("src/mapping/label_map.yaml")

ALLOWED = ["diatom", "protist", "detritus", "artifact", "other"]

def coarse_from_planktonset(raw: str) -> str:
    r = raw.lower()

    if r.startswith("diatom_"):
        return "diatom"

    if r.startswith(("protist_", "acantharia_", "radiolarian_")):
        return "protist"

    if r.startswith("detritus_") or r == "fecal_pellet":
        return "detritus"

    if r.startswith("artifacts"):
        return "artifact"

    if r.startswith("unknown_"):
        return "other"

    # Everything else is zooplankton-ish, but not shared with IFCB -> park in "other" for now
    return "other"

def main():
    if not ZOOSCAN_DIR.exists():
        raise SystemExit(f"Missing: {ZOOSCAN_DIR}")

    raw_classes = sorted([p.name for p in ZOOSCAN_DIR.iterdir() if p.is_dir()])
    if not raw_classes:
        raise SystemExit("No class folders found under data/zooscan/images")

    zooscan_map = {raw: coarse_from_planktonset(raw) for raw in raw_classes}

    # Load existing mapping if present, else start fresh
    if OUT.exists():
        with OUT.open("r", encoding="utf-8") as f:
            m = yaml.safe_load(f) or {}
    else:
        m = {}

    m.setdefault("ifcb", {})      # keep whatever you already generated for IFCB
    m["zooscan"] = zooscan_map    # overwrite zooscan mapping

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        yaml.safe_dump(m, f, sort_keys=True, allow_unicode=True)

    print(f"Wrote: {OUT}")
    print(f"Zooscan/PlanktonSet classes: {len(raw_classes)}")
    print("Example mappings:")
    for k in raw_classes[:20]:
        print(f"  {k} -> {zooscan_map[k]}")
    print("\nShared coarse labels:", ALLOWED)

if __name__ == "__main__":
    main()
