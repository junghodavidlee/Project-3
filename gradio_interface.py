"""
Gradio Interface for Sentiment Analysis Models
Provides interactive web interfaces for R2D2, Skywalker, and VADER models
"""

import gradio as gr
from models import R2D2Model, SkywalkerModel, VaderModel


class SentimentAnalysisInterface:
    """Create Gradio interfaces for sentiment analysis models"""

    @staticmethod
    def create_r2d2_interface(model):
        """Create Gradio interface for R2D2 model"""
        def predict(text):
            return model.predict(text)

        with gr.Blocks(css=".gradio-container {background-color: #266fd9; font-family: Arial;}") as demo:
            gr.Markdown("## R2D2 Sentiment Analysis")
            gr.Markdown("LSTM-based sentiment analysis model")

            with gr.Row():
                input_text = gr.Textbox(
                    label="Enter your review",
                    lines=3,
                    max_lines=5,
                    interactive=True,
                    placeholder="Type your review here..."
                )

            analyze_button = gr.Button("Analyze Sentiment")

            with gr.Row():
                output_label = gr.Textbox(label="Sentiment Result", interactive=False)

            analyze_button.click(predict, input_text, output_label)

        return demo

    @staticmethod
    def create_skywalker_interface(model):
        """Create Gradio interface for Skywalker model"""
        def predict(text):
            return model.predict(text)

        with gr.Blocks(css=".gradio-container {background-color: #3e8b4f; font-family: Arial;}") as demo:
            gr.Markdown("## Skywalker Sentiment Analysis")
            gr.Markdown("LinearSVC with TF-IDF vectorization")

            with gr.Row():
                input_text = gr.Textbox(
                    label="Enter your review",
                    lines=3,
                    max_lines=5,
                    interactive=True,
                    placeholder="Type your review here..."
                )

            analyze_button = gr.Button("Analyze Sentiment")

            with gr.Row():
                output_label = gr.Textbox(label="Sentiment Result", interactive=False)

            analyze_button.click(predict, input_text, output_label)

        return demo

    @staticmethod
    def create_vader_interface(model):
        """Create Gradio interface for VADER model"""
        def predict(text):
            sentiment, score = model.predict(text)
            return f"{sentiment} (Score: {score:.3f})"

        with gr.Blocks(css=".gradio-container {background-color: #df3120; font-family: Arial;}") as demo:
            gr.Markdown("## VADER Sentiment Analysis")
            gr.Markdown("Rule-based sentiment analysis using VADER")

            with gr.Row():
                input_text = gr.Textbox(
                    label="Enter your review",
                    lines=3,
                    max_lines=5,
                    interactive=True,
                    placeholder="Type your review here..."
                )

            analyze_button = gr.Button("Analyze Sentiment")

            with gr.Row():
                output_label = gr.Textbox(label="Sentiment Result", interactive=False)

            analyze_button.click(predict, input_text, output_label)

        return demo


def launch_all_interfaces(r2d2_model=None, skywalker_model=None, vader_model=None):
    """
    Launch all available sentiment analysis interfaces

    Args:
        r2d2_model: Trained R2D2 model
        skywalker_model: Trained Skywalker model
        vader_model: VADER model instance
    """
    interfaces = []

    if vader_model:
        vader_demo = SentimentAnalysisInterface.create_vader_interface(vader_model)
        interfaces.append(("VADER", vader_demo))

    if skywalker_model:
        skywalker_demo = SentimentAnalysisInterface.create_skywalker_interface(skywalker_model)
        interfaces.append(("Skywalker", skywalker_demo))

    if r2d2_model:
        r2d2_demo = SentimentAnalysisInterface.create_r2d2_interface(r2d2_model)
        interfaces.append(("R2D2", r2d2_demo))

    if interfaces:
        combined = gr.TabbedInterface(
            [demo for _, demo in interfaces],
            [name for name, _ in interfaces]
        )
        combined.launch(share=True)
