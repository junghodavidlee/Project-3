"""
Evaluation Script for Rotten Tomatoes Dataset
Test trained models on Rotten Tomatoes reviews
"""

import warnings
warnings.filterwarnings("ignore")

from data_loader import DataLoader
from models import SkywalkerModel, VaderModel
from evaluation import ModelEvaluator


def main():
    """Evaluate models on Rotten Tomatoes dataset"""
    print("=" * 60)
    print("Evaluating Models on Rotten Tomatoes Dataset")
    print("=" * 60)

    # Load Rotten Tomatoes dataset
    print("\nLoading Rotten Tomatoes dataset...")
    loader = DataLoader()
    rt_df = loader.load_rotten_tomatoes_dataset()
    print(f"Dataset loaded: {len(rt_df)} reviews")

    X_test = rt_df['text'].astype(str)
    y_test = rt_df['label']

    # Evaluate Skywalker model
    print("\n" + "=" * 60)
    print("Evaluating Skywalker Model")
    print("=" * 60)
    try:
        skywalker = SkywalkerModel()
        skywalker.load_model('vectorizer.pkl', 'model.pkl')

        predictions = skywalker.pipeline.predict(X_test)
        accuracy = skywalker.pipeline.score(X_test, y_test)

        print(f"\nSkywalker Accuracy on RT dataset: {accuracy:.4f}")

        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

    except Exception as e:
        print(f"Error evaluating Skywalker: {e}")

    # Evaluate VADER model
    print("\n" + "=" * 60)
    print("Evaluating VADER Model")
    print("=" * 60)

    vader = VaderModel()
    predictions = [vader.predict_binary(text) for text in X_test]

    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nVADER Accuracy on RT dataset: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
