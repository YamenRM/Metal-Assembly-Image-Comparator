import cv2
import numpy as np
import streamlit as st
from PIL import Image
from utils.alignment import align_images
from utils.comparator import compute_differences
from utils.visualizer import generate_comparison_view

st.set_page_config(page_title="AI Metal Assembly Image Comparator", layout="wide")
st.title("AI Metal Assembly Image Comparator")
st.caption("Upload a reference image and a target image to align them and highlight structural differences.")

# --- Session State Initialization ---
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

# --- UI Controls Layout ---
col1, col2 = st.columns(2)
with col1:
    img_a_file = st.file_uploader("Select Image A (Reference)", type=["jpg", "jpeg", "png", "bmp" , "gif"])
with col2:
    img_b_file = st.file_uploader("Select Image B (Target)", type=["jpg", "jpeg", "png", "bmp" , "gif"])

# --- Interactive Tuning Adjustments ---
st.subheader("🎛️ Fine-Tune Detection Parameters")

ssim_thresh = st.slider(
    "Detection Sensitivity (SSIM Threshold)", 
    min_value=0.10, 
    max_value=0.90, 
    value=0.45, 
    step=0.01,
    help="Higher sensitivity catches smaller structural variations. Lower sensitivity ignores surface scratches and dust."
)

min_area = st.slider(
    "Minimum Feature Size (Pixels)", 
    min_value=10, 
    max_value=1000, 
    value=50, 
    step=5,
    help="Increase this value to filter out small reflections and metallic glares on the edges of the structure."
)

# --- Process Execution ---
if st.button("Compare Images", type="primary"):
    if img_a_file is None or img_b_file is None:
        st.error("Please upload both images before comparing.")
    else:
        try:
            img_a = cv2.imdecode(np.frombuffer(img_a_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            img_b = cv2.imdecode(np.frombuffer(img_b_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

            if img_a is None or img_b is None:
                st.error("One or both uploaded files could not be read as images.")
            else:
                aligned_b, valid_mask = align_images(img_a, img_b)
                changes = compute_differences(
                    img_a,
                    aligned_b,
                    valid_mask=valid_mask,
                    ssim_thresh=ssim_thresh,
                    min_area=min_area,
                )
                composite = generate_comparison_view(img_a, aligned_b, changes)

                composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
                # Store the result in session state so it survives slider adjustments
                st.session_state.comparison_result = Image.fromarray(composite_rgb)
                st.success("Analysis complete!")
                
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

# --- Render Results Window ---
if st.session_state.comparison_result is not None:
    st.subheader("Comparison Result")
    st.image(
        st.session_state.comparison_result, 
        use_column_width=True, 
        caption="Reference image (left) and highlighted target image (right)"
    )
