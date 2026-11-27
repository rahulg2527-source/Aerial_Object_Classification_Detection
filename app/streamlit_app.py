
import sys
import io
import hashlib
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

# =========================================================
# PROJECT PATHS AND IMPORTS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.classification import ImageClassifier
from app.utils.detection import ObjectDetector


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Aerial AI | Bird & Drone Intelligence",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, #10243a, #174b70);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #d6e8f5;
        font-size: 15px;
        margin-bottom: 0;
    }

    .section-label {
        color: #4b91c3;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .info-card {
        padding: 20px;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 13px;
        min-height: 125px;
    }

    .info-card h3 {
        font-size: 15px;
        margin-bottom: 8px;
    }

    .info-card p {
        font-size: 13px;
        opacity: 0.75;
        margin-bottom: 0;
    }

    .footer {
        text-align: center;
        opacity: 0.6;
        font-size: 12px;
        padding-top: 25px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        padding: 15px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_classifier():
    return ImageClassifier()


@st.cache_resource
def load_detector():
    return ObjectDetector()


# =========================================================
# IMAGE SIGNATURE FUNCTION
# =========================================================

def get_image_signature(uploaded_file):
    image_bytes = uploaded_file.getvalue()

    return hashlib.sha256(image_bytes).hexdigest()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.title("🛩️ Aerial AI")

    st.caption("Bird & Drone Intelligence System")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Image Classification",
            "Object Detection",
            "Model Performance"
        ]
    )

    st.divider()

    st.markdown("### MODEL INFORMATION")

    st.write("**Classification:** MobileNetV2")
    st.write("**Detection:** YOLOv8n")
    st.write("**Classes:** Bird, Drone")

    st.divider()

    st.caption("AI-powered aerial image analysis")


# =========================================================
# REUSABLE PAGE HEADER
# =========================================================

def page_header(section, heading, description):

    st.markdown(
        f"""
        <div class="section-label">{section}</div>
        <h1>{heading}</h1>
        <p>{description}</p>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
    """
    <div class="hero">
    <div class="section-label">ARTIFICIAL INTELLIGENCE PROJECT</div>
    <h1>Aerial Object Classification &amp; Detection</h1>
    <p>An intelligent computer vision system designed to classify birds and detect birds and drones in aerial images.</p>
    </div>
    """,
        unsafe_allow_html=True
    )

    st.subheader("Project Overview")

    st.write(
        "This application combines two deep learning approaches "
        "to analyze aerial images. MobileNetV2 performs image-level "
        "classification, while YOLOv8 performs object-level detection."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="info-card">
            <h3>🧠 Image Classification</h3>
            <p>
                MobileNetV2 identifies whether an uploaded image
                contains a bird or a drone.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">
            <h3>🎯 Object Detection</h3>
            <p>
                YOLOv8 identifies objects and generates
                bounding boxes with confidence scores.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="info-card">
            <h3>📊 Model Evaluation</h3>
            <p>
                Performance metrics from the held-out test
                datasets are available for review.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Model Performance at a Glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Classification Accuracy", "98.14%")
    c2.metric("Detection Precision", "94.09%")
    c3.metric("Detection Recall", "85.72%")
    c4.metric("Detection mAP@50", "91.30%")

    st.caption(
        "Metrics are from previously evaluated held-out test sets. "
        "Classification and detection metrics measure different tasks."
    )

    st.divider()

    st.subheader("Technology Stack")

    st.write(
        "Python · PyTorch · MobileNetV2 · YOLOv8 · "
        "Ultralytics · OpenCV · Streamlit"
    )


# =========================================================
# IMAGE CLASSIFICATION
# =========================================================

elif page == "Image Classification":

    page_header(
        "AI MODULE 01",
        "Image Classification",
        "Classify an uploaded image as Bird or Drone using MobileNetV2."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="classification_file"
    )

    if uploaded_file is not None:

        # Generate a unique signature for the uploaded image
        image_signature = get_image_signature(uploaded_file)

        # Clear the old result when a different image is uploaded
        if st.session_state.get("classification_signature") != image_signature:

            st.session_state.pop("classification_result", None)

            st.session_state["classification_signature"] = image_signature

        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1.1, 1])

        with left:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with right:

            st.markdown("### Image Analysis")

            st.write(
                f"Image dimensions: {image.width} × {image.height}"
            )

            if st.button(
                "Run Classification",
                type="primary",
                use_container_width=True,
                key="classify_button"
            ):

                try:

                    with st.spinner("Analyzing image..."):

                        classifier = load_classifier()

                        result = classifier.predict(image)

                        st.session_state["classification_result"] = result

                        st.session_state["classification_signature"] = image_signature

                except Exception as error:

                    st.error(f"Classification failed: {error}")

        # Display results only for the currently uploaded image
        if (
            st.session_state.get("classification_signature") == image_signature
            and "classification_result" in st.session_state
        ):

            result = st.session_state["classification_result"]

            st.divider()

            st.subheader("Classification Results")

            c1, c2 = st.columns(2)

            c1.metric(
                "Predicted Class",
                result["class_name"].upper()
            )

            c2.metric(
                "Confidence Score",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.markdown("### Class Probability Distribution")

            probability_data = pd.DataFrame({
                "Class": [
                    name.title()
                    for name in result["probabilities"]
                ],
                "Probability (%)": [
                    value * 100
                    for value in result["probabilities"].values()
                ]
            })

            st.bar_chart(
                probability_data.set_index("Class")
            )

    else:

        # Clear the result when the uploaded file is removed
        st.session_state.pop("classification_result", None)
        st.session_state.pop("classification_signature", None)

        st.info("Upload an image to begin classification.")


# =========================================================
# OBJECT DETECTION
# =========================================================

elif page == "Object Detection":

    page_header(
        "AI MODULE 02",
        "Object Detection",
        "Detect birds and drones using the trained YOLOv8 model."
    )

    uploaded_file = st.file_uploader(
        "Upload an image for object detection",
        type=["jpg", "jpeg", "png"],
        key="detection_file"
    )

    if uploaded_file is not None:

        # Generate a unique signature for the uploaded image
        image_signature = get_image_signature(uploaded_file)

        # Clear the old result when a different image is uploaded
        if st.session_state.get("detection_signature") != image_signature:

            st.session_state.pop("detection_result", None)

            st.session_state["detection_signature"] = image_signature

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "Run Object Detection",
            type="primary",
            use_container_width=True,
            key="detect_button"
        ):

            try:

                with st.spinner("Detecting objects..."):

                    detector = load_detector()

                    result = detector.predict(image)

                    st.session_state["detection_result"] = result

                    st.session_state["detection_signature"] = image_signature

            except Exception as error:

                st.error(f"Detection failed: {error}")

        # Display results only for the currently uploaded image
        if (
            st.session_state.get("detection_signature") == image_signature
            and "detection_result" in st.session_state
        ):

            result = st.session_state["detection_result"]

            detections = result["detections"]

            st.divider()

            st.subheader("Detection Results")

            bird_count = sum(
                1 for item in detections
                if item["class_name"].lower() == "bird"
            )

            drone_count = sum(
                1 for item in detections
                if item["class_name"].lower() == "drone"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric("Total Objects", result["count"])
            c2.metric("Birds Detected", bird_count)
            c3.metric("Drones Detected", drone_count)

            st.markdown("### Annotated Image")

            st.image(
                result["annotated_image"],
                use_container_width=True
            )

            if detections:

                st.markdown("### Detection Details")

                details = pd.DataFrame([
                    {
                        "Object": item["class_name"].title(),
                        "Confidence (%)": round(
                            item["confidence"] * 100, 2
                        ),
                        "Bounding Box": str(item["box"])
                    }
                    for item in detections
                ])

                st.dataframe(
                    details,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning("No objects detected.")

            # Prepare annotated image for download
            buffer = io.BytesIO()

            result["annotated_image"].save(
                buffer,
                format="JPEG"
            )

            st.download_button(
                "⬇️ Download Annotated Image",
                data=buffer.getvalue(),
                file_name="aerial_detection_result.jpg",
                mime="image/jpeg",
                type="primary"
            )

    else:

        # Clear the result when the uploaded file is removed
        st.session_state.pop("detection_result", None)
        st.session_state.pop("detection_signature", None)

        st.info("Upload an image to begin object detection.")


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    page_header(
        "EVALUATION",
        "Model Performance",
        "Held-out test-set evaluation results for both models."
    )

    st.subheader("1. MobileNetV2 Classification")

    st.metric("Test Accuracy", "98.14%")

    classification_metrics = pd.DataFrame({
        "Class": ["Bird", "Drone"],
        "Precision (%)": [96.80, 100.00],
        "Recall (%)": [100.00, 95.74],
        "F1 Score (%)": [98.37, 97.83]
    })

    st.dataframe(
        classification_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("2. YOLOv8 Object Detection")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Precision", "94.09%")
    c2.metric("Recall", "85.72%")
    c3.metric("mAP@50", "91.30%")
    c4.metric("mAP@50-95", "58.69%")

    detection_metrics = pd.DataFrame({
        "Class": ["Bird", "Drone"],
        "Precision (%)": [94.90, 93.30],
        "Recall (%)": [88.60, 82.90],
        "mAP@50 (%)": [92.00, 90.60],
        "mAP@50-95 (%)": [61.90, 55.50]
    })

    st.dataframe(
        detection_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Evaluation Summary")

    st.write(
        "- MobileNetV2 achieved 98.14% accuracy on the classification test set."
    )

    st.write(
        "- YOLOv8 achieved 91.30% mAP@50 on the detection test set."
    )

    st.write(
        "- Detection recall is lower than precision, indicating that "
        "some annotated objects were missed."
    )

    st.write(
        "- The difference between mAP@50 and mAP@50-95 indicates "
        "that localization performance decreases under stricter "
        "intersection-over-union thresholds."
    )

    st.caption(
        "Classification and detection metrics are not directly comparable. "
        "Results are specific to the evaluated datasets."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
<div class="footer">
    Aerial Object Classification & Detection<br>
    Powered by MobileNetV2 · YOLOv8 · PyTorch · Streamlit
</div>
""", unsafe_allow_html=True)
