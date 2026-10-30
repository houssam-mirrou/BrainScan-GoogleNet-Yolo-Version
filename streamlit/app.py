import io
import os
import warnings

import requests
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 20_000_000

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="BrainScan AI", page_icon="🧠")
st.title("BrainScan AI")
st.warning("Research prototype. Not a clinically validated diagnostic tool.")
st.caption("No Tumor bounding boxes mark annotated non-tumor regions, not tumors.")

DETECTION_MODELS = {
    "unbalanced": "YOLOv8n — Unbalanced data (reference)",
    "balanced": "YOLOv8n — Balanced data",
}
detection_model = st.selectbox(
    "Detection model",
    options=list(DETECTION_MODELS),
    format_func=DETECTION_MODELS.get,
    help="Choose the YOLO training dataset. Both models detect the same four classes.",
)

uploaded = st.file_uploader("Upload an MRI image (up to 10 MB)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    content = uploaded.getvalue()
    if len(content) > 10 * 1024 * 1024:
        st.error("Image must be at most 10 MB.")
        st.stop()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        st.error("Please upload a valid JPG or PNG image.")
        st.stop()
    st.image(image, caption="Uploaded MRI", width="stretch")

    if st.button("Analyze", type="primary"):
        with st.spinner("Analyzing MRI…"):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, content, uploaded.type)},
                    data={"detection_model": detection_model},
                    timeout=120,
                )
                response.raise_for_status()
            except requests.RequestException as error:
                st.error(f"Analysis request failed: {error}")
                st.stop()
            result = response.json()

        classification = result["classification"]
        st.subheader(classification["predicted_class"])
        st.metric("Classification confidence", f"{classification['confidence']:.1%}")
        probabilities = classification["probabilities"]
        st.bar_chart(
            {"Class": list(probabilities), "Probability": list(probabilities.values())},
            x="Class", y="Probability",
        )

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for detection in result["detections"]:
            color = "green" if not detection["is_tumor"] else "red"
            box = detection["bbox"]
            draw.rectangle(box, outline=color, width=3)
            label = f"{detection['class_name']} {detection['confidence']:.1%}"
            draw.text((box[0], max(0, box[1] - 14)), label, fill=color)
        st.image(
            annotated,
            caption=f"Detections: {DETECTION_MODELS[result['detection_model']]}",
            width="stretch",
        )
        if result["detections"]:
            st.dataframe(result["detections"], hide_index=True)
        else:
            st.info("No bounding boxes detected. This does not establish absence of a tumor.")
