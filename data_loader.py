"""
Data Loading Module for Movie Sentiment Analysis
Handles loading and preprocessing of IMDB and Rotten Tomatoes datasets
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load and preprocess movie review datasets"""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def load_imdb_dataset(self, file_path='data/IMDB_Dataset.csv'):
        """
        Load IMDB dataset from CSV file

        Args:
            file_path (str): Path to IMDB CSV file

        Returns:
            pd.DataFrame: Loaded dataset
        """
        data = pd.read_csv(file_path)
        data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
        return data

    def load_rotten_tomatoes_dataset(self):
        """
        Load Rotten Tomatoes dataset from Hugging Face

        Returns:
            pd.DataFrame: Loaded dataset
        """
        rt = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        rt_df = rt['train'].to_pandas()
        return rt_df

    def prepare_data(self, data, text_col='review', label_col='sentiment', test_size=0.2):
        """
        Split data into training and test sets

        Args:
            data (pd.DataFrame): Input dataset
            text_col (str): Name of text column
            label_col (str): Name of label column
            test_size (float): Proportion of data for testing

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = data[text_col]
        y = data[label_col]
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)
