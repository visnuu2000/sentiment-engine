import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sentiment_model import load_model, predict

# 1. Initialize the model globally (happens when the script starts)
print("Loading model — please wait...")
classifier = load_model()
print("Model loaded!")

# 2. Define the 'Logic' function
def analyse_sentiment(user_input):
    if not user_input.strip():
        return "Please type something first.", "0%", None
    
    # Run prediction
    result = predict(user_input, classifier)
    
    # Prepare the Plot
    fig, ax = plt.subplots(figsize=(6, 2))
    colors = ["#1D9E75", "#E24B4A"]
    labels = ["Positive", "Negative"]
    
    if result["label"] == "Positive":
        vals = [result["confidence"], 100 - result["confidence"]]
    else:
        vals = [100 - result["confidence"], result["confidence"]]
        
    bars = ax.barh(labels, vals, color=colors, height=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    
    # Return outputs in the order they are defined in the Interface
    return result["label"], f"{result['confidence']}%", fig

# 3. Build the Gradio Interface
with gr.Blocks(title="Sentiment Analyser") as demo:
    gr.Markdown("# Sentiment Analysis Engine")
    gr.Markdown("Type any text below and find out if it is Positive or Negative.")
    
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                label="Enter your text here", 
                placeholder="Example: This product is absolutely amazing!",
                lines=5
            )
            btn = gr.Button("Analyse", variant="primary")
        
        with gr.Column():
            label_out = gr.Label(label="Sentiment")
            confidence_out = gr.Textbox(label="Confidence")
            plot_out = gr.Plot(label="Confidence Chart")

    # Connect the button to the function
    btn.click(
        fn=analyse_sentiment, 
        inputs=user_input, 
        outputs=[label_out, confidence_out, plot_out]
    )

# 4. Launch the app
if __name__ == "__main__":
    demo.launch()