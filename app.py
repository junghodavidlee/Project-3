"""
Gradio Application for Sentiment Analysis
Launch interactive web interface for all three models
"""

import warnings
warnings.filterwarnings("ignore")

from models import SkywalkerModel, VaderModel
from gradio_interface import launch_all_interfaces


def main():
    """Launch Gradio interface with all available models"""
    print("Loading models...")

    # Load Skywalker model (pre-trained)
    try:
        skywalker = SkywalkerModel()
        skywalker.load_model('vectorizer.pkl', 'model.pkl')
        print("✓ Skywalker model loaded")
    except Exception as e:
        print(f"⚠ Could not load Skywalker model: {e}")
        skywalker = None

    # Initialize VADER model
    vader = VaderModel()
    print("✓ VADER model initialized")

    # Note: R2D2 model is not included here as it requires the trained model to be saved
    # If you want to include R2D2, train it first and save it using model.save()

    print("\nLaunching Gradio interface...")
    launch_all_interfaces(
        r2d2_model=None,  # Add R2D2 model if available
        skywalker_model=skywalker,
        vader_model=vader
    )


if __name__ == "__main__":
    main()
