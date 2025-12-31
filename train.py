"""
Training Script for Sentiment Analysis Models
Train R2D2 (LSTM) and Skywalker (LinearSVC) models
"""

import warnings
warnings.filterwarnings("ignore")

from data_loader import DataLoader
from models import R2D2Model, SkywalkerModel, VaderModel
from evaluation import ModelEvaluator


def train_r2d2_model(X_train, X_test, y_train, y_test):
    """
    Train R2D2 LSTM model

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels

    Returns:
        R2D2Model: Trained model
    """
    print("=" * 60)
    print("Training R2D2 (LSTM) Model")
    print("=" * 60)

    r2d2 = R2D2Model(max_vocab_size=10000, max_sequence_length=200, embedding_dim=100)
    history = r2d2.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)

    print("\nEvaluating R2D2 model on test set...")
    X_train_pad, X_test_pad = r2d2.prepare_data(X_train, X_test)
    metrics = ModelEvaluator.evaluate_model(r2d2.model, X_test_pad, y_test)
    ModelEvaluator.print_evaluation(metrics)

    return r2d2


def train_skywalker_model(X_train, X_test, y_train, y_test):
    """
    Train Skywalker LinearSVC model

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels

    Returns:
        SkywalkerModel: Trained model
    """
    print("\n" + "=" * 60)
    print("Training Skywalker (LinearSVC + TF-IDF) Model")
    print("=" * 60)

    skywalker = SkywalkerModel()
    skywalker.train(X_train, y_train)

    print("\nEvaluating Skywalker model...")
    print(f"Train accuracy: {skywalker.pipeline.score(X_train, y_train):.4f}")
    print(f"Test accuracy: {skywalker.pipeline.score(X_test, y_test):.4f}")

    predictions = skywalker.pipeline.predict(X_test)
    metrics = {
        'accuracy': skywalker.pipeline.score(X_test, y_test),
        'confusion_matrix': None,
        'classification_report': None
    }

    from sklearn.metrics import confusion_matrix, classification_report
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nSaving model...")
    skywalker.save_model()
    print("Model saved to vectorizer.pkl and model.pkl")

    return skywalker


def evaluate_vader_model(X_test, y_test):
    """
    Evaluate VADER model

    Args:
        X_test: Test features
        y_test: Test labels
    """
    print("\n" + "=" * 60)
    print("Evaluating VADER Model")
    print("=" * 60)

    vader = VaderModel()

    predictions = [vader.predict_binary(text) for text in X_test]

    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return vader


def main():
    """Main training pipeline"""
    print("Movie Sentiment Analysis - Training Pipeline")
    print("=" * 60)

    # Load data
    print("\nLoading IMDB dataset...")
    loader = DataLoader(random_state=42)
    data = loader.load_imdb_dataset('data/IMDB_Dataset.csv')
    print(f"Dataset loaded: {len(data)} reviews")

    # Prepare data
    X_train, X_test, y_train, y_test = loader.prepare_data(data)
    print(f"Train set: {len(X_train)} reviews")
    print(f"Test set: {len(X_test)} reviews")

    # Train R2D2 model
    r2d2 = train_r2d2_model(X_train, X_test, y_train, y_test)

    # Train Skywalker model
    skywalker = train_skywalker_model(X_train, X_test, y_train, y_test)

    # Evaluate VADER model
    vader = evaluate_vader_model(X_test, y_test)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python app.py' to launch the Gradio interface")
    print("2. Run 'python evaluate_on_rt.py' to test on Rotten Tomatoes dataset")


if __name__ == "__main__":
    main()
