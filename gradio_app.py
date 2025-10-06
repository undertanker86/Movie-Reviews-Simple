import os
import joblib
import gradio as gr
import numpy as np
from typing import List, Tuple


def load_model():
    """Load the trained sentiment analysis model"""
    model_path_candidates = [
        os.path.join("model", "logistic_regression_sentiment_model.joblib"),
        "logistic_regression_sentiment_model.joblib",
        os.path.join("model", "logreg_pipeline.joblib"),
        "logreg_pipeline.joblib",
        os.path.join("model", "logistic_regression_pipeline.joblib"),
        "logistic_regression_pipeline.joblib",
    ]
    for path in model_path_candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(
        "Model file not found. Please ensure the model file exists in the 'model/' directory."
    )


def predict_sentiment(text: str) -> Tuple[str, float]:
    """
    Predict sentiment for a single text input
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (sentiment_label, confidence_score)
    """
    if not text.strip():
        return "Please enter some text to analyze.", 0.0
    
    try:
        # Get prediction
        prediction = model.predict([text])[0]
        
        # Get probability if available
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            if len(proba) >= 2:
                confidence = float(proba[1])  # Positive sentiment probability
        
        # Convert prediction to label
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        return sentiment, confidence
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0


def predict_batch(texts: str) -> List[Tuple[str, float]]:
    """
    Predict sentiment for multiple texts (one per line)
    
    Args:
        texts: Multiple texts separated by newlines
        
    Returns:
        List of tuples (sentiment_label, confidence_score)
    """
    if not texts.strip():
        return [("Please enter some text to analyze.", 0.0)]
    
    text_list = [line.strip() for line in texts.split('\n') if line.strip()]
    if not text_list:
        return [("Please enter some text to analyze.", 0.0)]
    
    try:
        # Get predictions
        predictions = model.predict(text_list)
        
        # Get probabilities if available
        confidences = [0.0] * len(text_list)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(text_list)
            if probas.shape[1] >= 2:
                confidences = probas[:, 1].astype(float).tolist()
        
        # Convert predictions to labels
        results = []
        for pred, conf in zip(predictions, confidences):
            sentiment = "Positive" if pred == 1 else "Negative"
            results.append((sentiment, conf))
        
        return results
        
    except Exception as e:
        return [(f"Error: {str(e)}", 0.0)]


# Load the model
try:
    model = load_model()
    model_loaded = True
    model_status = "✅ Model loaded successfully!"
except Exception as e:
    model_loaded = False
    model_status = f"❌ Model loading failed: {str(e)}"


# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="IMDB Sentiment Analysis",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .status-box {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .status-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .status-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        """
    ) as demo:
        
        gr.HTML(f"""
        <div class="main-header">
            <h1>🎬 IMDB Movie Review Sentiment Analysis</h1>
            <p>Analyze the sentiment of movie reviews using machine learning</p>
        </div>
        """)
        
        # Model status
        status_class = "status-success" if model_loaded else "status-error"
        gr.HTML(f"""
        <div class="status-box {status_class}">
            <strong>Model Status:</strong> {model_status}
        </div>
        """)
        
        if model_loaded:
            with gr.Tabs():
                # Single text prediction tab
                with gr.Tab("Single Review Analysis"):
                    gr.Markdown("### Enter a movie review to analyze its sentiment")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            single_input = gr.Textbox(
                                label="Movie Review Text",
                                placeholder="Enter your movie review here...",
                                lines=5,
                                max_lines=10
                            )
                            single_btn = gr.Button("Analyze Sentiment", variant="primary")
                        
                        with gr.Column(scale=1):
                            single_output = gr.Textbox(
                                label="Sentiment",
                                interactive=False,
                                lines=2
                            )
                            single_confidence = gr.Number(
                                label="Confidence Score",
                                interactive=False,
                                precision=3
                            )
                    
                    single_btn.click(
                        fn=predict_sentiment,
                        inputs=single_input,
                        outputs=[single_output, single_confidence]
                    )
                
                # Batch prediction tab
                with gr.Tab("Batch Analysis"):
                    gr.Markdown("### Analyze multiple reviews at once (one per line)")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            batch_input = gr.Textbox(
                                label="Multiple Reviews (one per line)",
                                placeholder="Enter multiple reviews, one per line...",
                                lines=8,
                                max_lines=15
                            )
                            batch_btn = gr.Button("Analyze All Reviews", variant="primary")
                        
                        with gr.Column(scale=1):
                            batch_output = gr.Dataframe(
                                headers=["Review", "Sentiment", "Confidence"],
                                datatype=["str", "str", "number"],
                                interactive=False,
                                wrap=True
                            )
                    
                    def format_batch_results(texts: str) -> List[List]:
                        """Format batch results for display"""
                        text_list = [line.strip() for line in texts.split('\n') if line.strip()]
                        if not text_list:
                            return []
                        
                        results = predict_batch(texts)
                        formatted_results = []
                        for i, (sentiment, confidence) in enumerate(results):
                            # Truncate long reviews for display
                            display_text = text_list[i][:100] + "..." if len(text_list[i]) > 100 else text_list[i]
                            formatted_results.append([display_text, sentiment, f"{confidence:.3f}"])
                        
                        return formatted_results
                    
                    batch_btn.click(
                        fn=format_batch_results,
                        inputs=batch_input,
                        outputs=batch_output
                    )
                
                # Examples tab
                with gr.Tab("Examples"):
                    gr.Markdown("### Try these example reviews")
                    
                    examples = [
                        "This movie is absolutely fantastic! The acting was superb and the plot was engaging from start to finish.",
                        "I was really disappointed with this film. The story was boring and the characters were poorly developed.",
                        "An amazing cinematic experience! The special effects were incredible and the soundtrack was perfect.",
                        "This is one of the worst movies I've ever seen. Terrible acting and a confusing plot.",
                        "A decent movie with some good moments, but overall it was just okay. Nothing special.",
                        "Outstanding performance by the lead actor! This movie exceeded all my expectations.",
                        "The movie had potential but fell flat. The pacing was too slow and the ending was unsatisfying.",
                        "Brilliant direction and cinematography! This is a must-watch for any film enthusiast."
                    ]
                    
                    gr.Examples(
                        examples=examples,
                        inputs=single_input,
                        label="Click on any example to try it"
                    )
        
        else:
            gr.Markdown("""
            ### ⚠️ Model Not Available
            
            The sentiment analysis model could not be loaded. Please check that:
            - The model file exists in the `model/` directory
            - The model file is properly trained and saved
            - All dependencies are installed correctly
            """)
    
    return demo


if __name__ == "__main__":
    print("Starting Gradio interface...")
    print("The interface will be available at: http://localhost:7860")
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
