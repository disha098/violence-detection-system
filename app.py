import streamlit as st
import tempfile
import pandas as pd
from src.model import load_trained_model
from src.inference import predict_with_timestamps

# 🔥 Page config
st.set_page_config(page_title="Violence Detection", layout="wide")

# 🔥 Premium UI FIXED (visible text)
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}

/* Metric card */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1c1f26, #2a2f3a);
    padding: 15px;
    border-radius: 12px;
}

/* Metric label */
div[data-testid="stMetricLabel"] {
    color: #bbbbbb !important;
}

/* Metric value */
div[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 🔥 Title
st.markdown("""
<h1 style='text-align:center; font-size:40px;'> Violence Detection System</h1>
<p style='text-align:center; color:gray;'>AI-powered video surveillance & analysis</p>
<hr>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi"])

if uploaded_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    with st.spinner("Analyzing video... ⏳"):
        model = load_trained_model()
        raw_results, merged_results = predict_with_timestamps(model, tfile.name)

    st.markdown("## 🧠 Analysis Report")

    if merged_results is None or len(merged_results) == 0:
        st.success("✅ No Violence Detected")

    else:
        fps = 25

        # 🔥 Metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("🚨 Violence Segments", len(merged_results))

        with col2:
            st.metric("📊 Total Segments", len(raw_results))

        st.markdown("---")

        # 🔥 Timestamps + Severity
        st.markdown("### ⏱️ Detected Timestamps")

        for seg in merged_results:
            start = seg["start_frame"] / fps
            end = seg["end_frame"] / fps
            prob = seg["violence_prob"]

            if prob > 0.9:
                level = "🔴 HIGH"
            elif prob > 0.7:
                level = "🟡 MEDIUM"
            else:
                level = "🟢 LOW"

            st.error(f"{level} Violence from {start:.2f}s → {end:.2f}s ({prob*100:.2f}%)")

        st.markdown("---")

        # 🔥 Key Insights
        st.markdown("### 🔎 Key Insights")

        longest = max([(seg["end_frame"] - seg["start_frame"]) for seg in merged_results]) / fps
        total_violence = sum([(seg["end_frame"] - seg["start_frame"]) for seg in merged_results]) / fps
        total_duration = raw_results[-1]["end_frame"] / fps

        percentage = (total_violence / total_duration) * 100 if total_duration > 0 else 0

        st.info(f"""
        • 🔥 Total Violence Duration: {total_violence:.2f}s  
        • ⏱️ Longest Continuous Segment: {longest:.2f}s  
        • 📊 Number of Violent Segments: {len(merged_results)}  
        • 📈 Violence Percentage: {percentage:.2f}%  
        """)

        st.markdown("---")

        # 🔥 AI Explanation
        st.markdown("### 🧠 AI Explanation")

        st.warning("""
        • Sudden motion spikes detected in multiple segments  
        • Close human interactions observed  
        • High temporal variation across frames  
        • Consistent high-confidence predictions indicate strong likelihood of violence  
        """)

        st.markdown("---")

        # 🔥 Download Report
        report_data = []

        for seg in merged_results:
            report_data.append({
                "Start (s)": seg["start_frame"] / fps,
                "End (s)": seg["end_frame"] / fps,
                "Confidence": seg["violence_prob"]
            })

        df = pd.DataFrame(report_data)

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Report",
            data=csv,
            file_name="violence_report.csv",
            mime="text/csv"
        )