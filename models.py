"""
Machine Learning Models for Sentiment Analysis
Contains R2D2 (LSTM), Skywalker (LinearSVC), and VADER implementations
"""

import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class R2D2Model:
    """LSTM-based sentiment analysis model"""

    def __init__(self, max_vocab_size=10000, max_sequence_length=200, embedding_dim=100):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None

    def build_model(self):
        """Build LSTM model architecture"""
        self.model = Sequential([
            Embedding(input_dim=self.max_vocab_size,
                     output_dim=self.embedding_dim,
                     input_length=self.max_sequence_length),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return self.model

    def prepare_data(self, X_train, X_test):
        """Tokenize and pad sequences"""
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(X_train)

        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)

        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_sequence_length,
                                   padding="post", truncating="post")
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_sequence_length,
                                  padding="post", truncating="post")

        return X_train_pad, X_test_pad

    def train(self, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
        """Train the LSTM model"""
        X_train_pad, X_test_pad = self.prepare_data(X_train, X_test)
        if self.model is None:
            self.build_model()

        history = self.model.fit(X_train_pad, y_train,
                               epochs=epochs,
                               batch_size=batch_size,
                               validation_data=(X_test_pad, y_test))
        return history

    def predict(self, text):
        """Predict sentiment for input text"""
        if isinstance(text, str):
            text = [text]

        seq = self.tokenizer.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=self.max_sequence_length, padding="post")
        prediction = self.model.predict(padded)

        return "Positive" if prediction[0][0] > 0.5 else "Negative"


class SkywalkerModel:
    """LinearSVC with TF-IDF vectorization"""

    def __init__(self):
        self.custom_stopwords = [
            'a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by',
            'can', 'even', 'ever', 'for', 'from', 'get', 'had', 'has', 'have', 'he',
            'her', 'hers', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its',
            'just', 'me', 'my', 'of', 'on', 'or', 'see', 'seen', 'she', 'so', 'than',
            'that', 'the', 'their', 'there', 'they', 'this', 'to', 'was', 'we', 'were',
            'what', 'when', 'which', 'who', 'will', 'with', 'you'
        ]
        self.pipeline = None

    def build_pipeline(self):
        """Build TF-IDF + LinearSVC pipeline"""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words=self.custom_stopwords,
                ngram_range=(1, 2),
                max_df=0.9,
                min_df=2,
                sublinear_tf=True
            )),
            ('clf', LinearSVC(C=1, loss='squared_hinge', penalty='l2', dual=True))
        ])
        return self.pipeline

    def train(self, X_train, y_train):
        """Train the model"""
        if self.pipeline is None:
            self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

    def predict(self, text):
        """Predict sentiment for input text"""
        if isinstance(text, str):
            text = [text]

        prediction = self.pipeline.predict(text)[0]
        return "😊 Positive" if prediction == 1 else "😠 Negative"

    def save_model(self, vectorizer_path='vectorizer.pkl', model_path='model.pkl'):
        """Save the trained model"""
        vectorizer = self.pipeline.named_steps['tfidf']
        model = self.pipeline.named_steps['clf']
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(model, model_path)

    def load_model(self, vectorizer_path='vectorizer.pkl', model_path='model.pkl'):
        """Load a saved model"""
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        self.pipeline = Pipeline([('tfidf', vectorizer), ('clf', model)])


class VaderModel:
    """VADER sentiment analysis model"""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text):
        """Predict sentiment with score"""
        score = self.analyzer.polarity_scores(text)['compound']

        if score >= 0.05:
            sentiment = "😊 Positive"
        elif score <= -0.05:
            sentiment = "😠 Negative"
        else:
            sentiment = "😐 Neutral"

        return sentiment, score

    def predict_binary(self, text):
        """Predict binary sentiment (0 or 1)"""
        score = self.analyzer.polarity_scores(str(text))["compound"]
        return 1 if score >= 0 else 0
