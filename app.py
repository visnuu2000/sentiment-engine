import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sentiment_model import load_model, predict

st.set_page_config(page_title="Sentiment Analyser", layout="centered")
st.title("Sentiment Analysis Engine")
st.write("Type any text below and find out if it is Positive or Negative.")

with st.spinner("Loading model — please wait about 30 seconds on first run..."):
    classifier = load_model()

st.success("Model loaded. Ready to analyse!")

user_input = st.text_area(
    "Enter your text here",
    placeholder="Example: This product is absolutely amazing and works perfectly!",
    height=150
)

if st.button("Analyse"):
    if not user_input.strip():
        st.warning("Please type something first.")
    else:
        result = predict(user_input, classifier)
        st.subheader("Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", result["label"])
        with col2:
            st.metric("Confidence", str(result["confidence"]) + "%")

        fig, ax = plt.subplots(figsize=(5, 2))
        colors = ["#1D9E75", "#E24B4A"]
        labels = ["Positive", "Negative"]
        if result["label"] == "Positive":
            vals = [result["confidence"], 100 - result["confidence"]]
        else:
            vals = [100 - result["confidence"], result["confidence"]]
        bars = ax.barh(labels, vals, color=colors, height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)")
        ax.bar_label(bars, fmt="%.1f%%", padding=4)
        ax.spines[["top", "right", "left"]].set_visible(False)
        st.pyplot(fig)
        plt.close()