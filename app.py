import streamlit as st
import tempfile
from src.model import load_trained_model
from src.inference import predict_with_timestamps

st.set_page_config(page_title="Violence Detection", layout="wide")

st.markdown("<h1 style='text-align:center;'>🔥 Violence Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi"])

if uploaded_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    with st.spinner("Analyzing video..."):
        model = load_trained_model()
        results = predict_with_timestamps(model, tfile.name)

    st.markdown("## 🧠 Analysis Report")

    if results is None:
        st.error("Error processing video")
    else:

        violence_segments = [r for r in results if r["violence_prob"] > 0.5]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🚨 Violence Segments", len(violence_segments))

        with col2:
            st.metric("📊 Total Segments", len(results))

        st.markdown("---")

        st.markdown("### ⏱️ Detected Timestamps")

        for seg in violence_segments:
            start = seg["start_frame"]
            end = seg["end_frame"]
            prob = seg["violence_prob"]

            st.warning(f"⚠️ Violence from frame {start} to {end} (Confidence: {prob*100:.2f}%)")

        st.markdown("---")

        st.markdown("### 🧠 AI Explanation")

        if len(violence_segments) > 0:
            st.info("""
            The system detected aggressive motion patterns and human interactions 
            consistent with violent behavior. High temporal changes and sudden 
            movements contributed to the classification.
            """)
        else:
            st.success("""
            No significant aggressive motion patterns were detected. 
            The scene appears safe with normal activity.
            """)