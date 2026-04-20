from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return classifier

def predict(text, classifier):
    if not text.strip():
        return None
    result = classifier(text[:512])[0]
    label = result["label"]
    score = round(result["score"] * 100, 1)
    if label == "POSITIVE":
        display = "Positive"
    else:
        display = "Negative"
    return {"label": display, "confidence": score}