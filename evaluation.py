"""
Model Evaluation Module
Provides evaluation metrics and visualization functions
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    cross_val_score
)


class ModelEvaluator:
    """Evaluate sentiment analysis models"""

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Evaluate model performance

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            dict: Evaluation metrics
        """
        y_pred = model.predict(X_test)

        if hasattr(y_pred[0], '__len__'):
            y_pred = (y_pred > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }

    @staticmethod
    def print_evaluation(metrics):
        """Print evaluation metrics in formatted way"""
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])

    @staticmethod
    def cross_validate(model, X, y, cv=5):
        """
        Perform cross-validation

        Args:
            model: Model to validate
            X: Features
            y: Labels
            cv (int): Number of folds

        Returns:
            dict: Cross-validation scores
        """
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        return {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }

    @staticmethod
    def print_cross_validation(cv_results):
        """Print cross-validation results"""
        print(f"Cross-validation scores: {cv_results['scores']}")
        print(f"Mean accuracy: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
