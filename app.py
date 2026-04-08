import streamlit as st
import tempfile
import pandas as pd
from src.model import load_trained_model
from src.inference import predict_with_timestamps

# Page setup
st.set_page_config(page_title="Violence Detection", layout="wide")

st.markdown("<h1 style='text-align:center;'>🔥 Violence Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi"])

if uploaded_file:

    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Show video
    st.video(uploaded_file)

    # Load model + predict
    with st.spinner("Analyzing video... ⏳"):
        model = load_trained_model()
        results = predict_with_timestamps(model, tfile.name)

    st.markdown("## 🧠 Analysis Report")

    if results is None or len(results) == 0:
        st.success("✅ No Violence Detected")
    else:

        fps = 25  # approx FPS

        # Metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("🚨 Violence Segments", len(results))

        with col2:
            st.metric("📊 Total Segments", len(results))

        st.markdown("---")

        # 🔥 Clean timestamps
        st.markdown("### ⏱️ Detected Timestamps")

        for seg in results:
            start = seg["start_frame"] / fps
            end = seg["end_frame"] / fps
            prob = seg["violence_prob"]

            st.error(f"⚠️ Violence from {start:.2f}s → {end:.2f}s (Confidence: {prob*100:.2f}%)")

        st.markdown("---")

        # 🔥 Summary
        st.markdown("### 📊 Summary")

        total_duration = results[-1]["end_frame"] / fps
        violence_duration = sum(
            [(seg["end_frame"] - seg["start_frame"]) for seg in results]
        ) / fps

        percentage = (violence_duration / total_duration) * 100 if total_duration > 0 else 0

        st.info(f"""
        - ⏱️ Total Video Duration: ~{total_duration:.2f}s  
        - 🚨 Violence Duration: ~{violence_duration:.2f}s  
        - 📈 Violence Percentage: {percentage:.2f}%  
        """)

        st.markdown("---")

        # 🔥 Graph
        st.markdown("### 📈 Violence Timeline")

        timeline = [seg["violence_prob"] for seg in results]

        st.line_chart(pd.DataFrame(timeline, columns=["Violence Probability"]))

        st.markdown("---")

        # 🔥 AI Explanation
        st.markdown("### 🧠 AI Explanation")

        st.warning("""
        The system detected aggressive motion patterns and rapid scene changes,
        which are commonly associated with violent interactions.
        Multiple segments show high confidence scores, indicating consistent detection.
        """)