# === file: app/streamlit_app.py ===
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH so `import src...` works in Streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2

from src.explainability import grad_cam_heatmap
from src.traits import segment_simple, compute_traits, overlay_mask

st.set_page_config(page_title="PlanktonShift Demo", layout="wide")


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def pil_to_float_rgb(img: Image.Image, size: int = 224) -> np.ndarray:
    img = img.convert("RGB").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def main():
    st.title("PlanktonShift: Cross-Instrument Plankton ML (Demo)")

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model path", value="models/ifcb_to_planktonset_coral.keras")
        image_size = st.number_input("Image size", 64, 512, 224, 32)
        st.caption("Tip: point to a saved .keras model file from training.")

    model = None
    try:
        model = load_model(model_path)
    except Exception as e:
        st.warning(f"Could not load model at {model_path}. Error: {e}")

    tab1, tab2, tab3 = st.tabs(["1) Predict + Grad-CAM", "2) Traits", "3) Cross-Domain Benchmarks"])

    with tab1:
        st.subheader("Upload image → predicted class + confidence + Grad-CAM")
        up = st.file_uploader("Upload a plankton image", type=["png", "jpg", "jpeg", "webp"], key="pred_uploader")
        if up and model is not None:
            img_pil = Image.open(up)
            colA, colB = st.columns([1, 1])

            x = pil_to_float_rgb(img_pil, size=int(image_size))
            logits = model(x[None, ...], training=False).numpy()[0]
            probs = softmax(logits)

            topk = probs.argsort()[::-1][:5]
            df = pd.DataFrame({"class_id": topk, "prob": probs[topk]})

            with colA:
                st.image(img_pil, caption="Input", use_container_width=True)
                st.dataframe(df, use_container_width=True)

            pred_class = int(topk[0])
            heat = grad_cam_heatmap(model, x, class_index=pred_class)
            heat_u8 = (heat * 255).clip(0, 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

            base_u8 = (x * 255).clip(0, 255).astype(np.uint8)
            overlay = cv2.addWeighted(base_u8, 0.6, heat_color, 0.4, 0)

            with colB:
                st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

    with tab2:
        st.subheader("Segmentation overlay → ESD/area/biovolume-proxy → CSV export")
        up2 = st.file_uploader("Upload image for trait extraction", type=["png", "jpg", "jpeg", "webp"], key="traits_uploader")
        if up2:
            img_pil = Image.open(up2)
            x = pil_to_float_rgb(img_pil, size=int(image_size))

            mask = segment_simple((x * 255).astype(np.uint8))
            traits = compute_traits(mask)
            over = overlay_mask((x * 255).astype(np.uint8), mask)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(img_pil, caption="Input", use_container_width=True)
            with c2:
                st.image(over, caption="Segmentation overlay (simple)", use_container_width=True)

            tdf = pd.DataFrame([traits.__dict__])
            st.dataframe(tdf, use_container_width=True)

            csv = tdf.to_csv(index=False).encode("utf-8")
            st.download_button("Download traits CSV", csv, file_name="traits.csv", mime="text/csv")

    with tab3:
        st.subheader("Load precomputed evaluation artifacts (fast demo)")
        st.caption("Point to JSON outputs from src/eval.py and src/calibrate.py.")

        eval_path = st.text_input("Eval JSON path", value="reports/artifacts/eval_planktonset.json")
        cal_path = st.text_input("Calibration JSON path", value="reports/artifacts/cal_planktonset.json")

        if Path(eval_path).exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                ev = json.load(f)
            st.markdown(
                f"**Dataset:** {ev.get('dataset')}  \n"
                f"**N:** {ev.get('n')}  \n"
                f"**ECE:** {ev.get('ece'):.4f}  \n"
                f"**Brier:** {ev.get('brier'):.4f}"
            )
            rep = ev.get("report", {})
            if "macro avg" in rep:
                st.write("Macro avg:")
                st.json(rep["macro avg"])
            st.write("Confusion matrix:")
            st.dataframe(pd.DataFrame(ev.get("confusion_matrix")), use_container_width=True)
        else:
            st.info("Eval JSON not found yet. Run: python -m src.eval ...")

        if Path(cal_path).exists():
            with open(cal_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
            st.markdown(
                f"**Temperature (T):** {cal.get('T')}  \n"
                f"**ECE:** {cal.get('ece_before'):.4f} → {cal.get('ece_after'):.4f}  \n"
                f"**Brier:** {cal.get('brier_before'):.4f} → {cal.get('brier_after'):.4f}"
            )
        else:
            st.info("Calibration JSON not found yet. Run: python -m src.calibrate ...")


if __name__ == "__main__":
    main()
