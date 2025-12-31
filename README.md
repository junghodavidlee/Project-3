 <img src= "https://legacybox.com/cdn/shop/articles/LBFilmReel_991x.progressive.jpg?v=1563402798" alt="Alt Text" width="800" height="350">

# Movie Sentiment Analysis
## Jungho Lee, James Kim, Maddie Jankowski

### Project Overview

**Background:**
Movie review sites can have hundreds of thousands of entries with different sentiments. Automated sentiment analysis helps users quickly understand the overall reception of movies without reading through countless reviews.

**Aim:**
We created multiple machine learning models to automatically provide sentiment analysis for movie reviews, targeting users who want to get metadata about a set of reviews.

### Dataset Overview
We used two datasets for training and evaluation:
- **IMDB Dataset**: 50,000 movie reviews with binary sentiment labels (positive/negative)
- **Rotten Tomatoes Dataset**: 8,530 movie reviews from the Cornell Movie Review dataset

The target column of both datasets is a positive or negative label of the review.

### Project Structure

```
Project-3/
├── data_loader.py          # Data loading and preprocessing utilities
├── models.py               # Model implementations (R2D2, Skywalker, VADER)
├── evaluation.py           # Model evaluation and metrics
├── gradio_interface.py     # Gradio web interface components
├── train.py                # Training script for all models
├── app.py                  # Launch Gradio application
├── evaluate_on_rt.py       # Evaluate models on Rotten Tomatoes dataset
├── requirements.txt        # Python dependencies
├── model.pkl               # Saved Skywalker model
├── vectorizer.pkl          # Saved TF-IDF vectorizer
├── notebooks/
│   ├── IMDB_Movie_Sentiment_Analysis.ipynb  # Jupyter notebook with model development
│   └── archive/            # Archived development notebooks
└── data/
    └── IMDB_Dataset.csv    # IMDB dataset (download required)
```

### Models

#### 1. R2D2 (LSTM Neural Network)
- Architecture: Bidirectional LSTM with embedding layer
- Preprocessing: Tokenization and sequence padding
- Performance: 87% accuracy on IMDB, 73% on Rotten Tomatoes

#### 2. Skywalker (LinearSVC + TF-IDF)
- Architecture: Linear Support Vector Classifier with TF-IDF vectorization
- Features: Custom stopwords, bigrams, sublinear TF scaling
- Performance: 92% accuracy on IMDB, 75% on Rotten Tomatoes

#### 3. VADER (Rule-based)
- Architecture: Valence Aware Dictionary and sEntiment Reasoner
- Approach: Lexicon and rule-based sentiment analysis
- Performance: 70% accuracy on IMDB, 61% on Rotten Tomatoes

### Results Summary

| Model      | IMDB Accuracy | Rotten Tomatoes Accuracy |
|------------|---------------|--------------------------|
| Skywalker  | 92%          | 75%                      |
| R2D2       | 87%          | 73%                      |
| VADER      | 70%          | 61%                      |

Although Skywalker had the best performance on the IMDB dataset, all models showed decreased accuracy on the Rotten Tomatoes dataset. This suggests domain differences between the datasets that could be explored in future work.

### Development Notebooks

The `notebooks/` directory contains Jupyter notebooks used during model development and experimentation:
- **IMDB_Movie_Sentiment_Analysis.ipynb**: Complete notebook showing the development process for all three models (R2D2, Skywalker, and VADER), including training, evaluation, and Gradio interface prototypes

The production code has been refactored into modular Python scripts for better maintainability and deployment.

### Installation & Usage

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Installation

1. Clone the repository:
```bash
git clone https://github.com/junghodavidlee/Project-3.git
cd Project-3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the IMDB dataset and place it in the `data/` directory:
   - Create a `data/` folder if it doesn't exist
   - Download IMDB_Dataset.csv and place it in `data/IMDB_Dataset.csv`

4. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

#### Training Models

Train all models from scratch:
```bash
python train.py
```

This will:
- Train the R2D2 LSTM model
- Train the Skywalker LinearSVC model
- Evaluate the VADER model
- Save trained models to disk

#### Running the Web Interface

Launch the interactive Gradio interface:
```bash
python app.py
```

This will start a web server with tabs for each model where you can:
- Enter movie reviews
- Get real-time sentiment predictions
- Compare results across different models

#### Evaluating on Rotten Tomatoes

Test the trained models on the Rotten Tomatoes dataset:
```bash
python evaluate_on_rt.py
```

### API Usage

You can also use the models programmatically:

```python
from models import SkywalkerModel, VaderModel
from data_loader import DataLoader

# Load and prepare data
loader = DataLoader()
data = loader.load_imdb_dataset('data/IMDB_Dataset.csv')
X_train, X_test, y_train, y_test = loader.prepare_data(data)

# Train Skywalker model
skywalker = SkywalkerModel()
skywalker.train(X_train, y_train)

# Make predictions
review = "This movie was absolutely amazing!"
sentiment = skywalker.predict(review)
print(sentiment)  # Output: "😊 Positive"

# Use VADER
vader = VaderModel()
sentiment, score = vader.predict(review)
print(f"{sentiment} (Score: {score})")
```

### Approach & Methodology

This project compares the performance of machine learning models (R2D2, Skywalker) with the rule-based VADER sentiment classifier.

**Attempted approaches:**
- Named Entity Recognition (NER) using spaCy to analyze whether the number of entities (PERSON, etc.) correlates with sentiment
- Results were inconclusive, suggesting entity count is not a strong predictor of sentiment

### Future Work

- Investigate why models perform worse on Rotten Tomatoes dataset
- Experiment with transformer-based models (BERT, RoBERTa)
- Implement multi-class sentiment analysis (positive, negative, neutral)
- Add aspect-based sentiment analysis to identify what aspects of movies are praised/criticized
- Expand to other review domains (books, restaurants, products)

### Conclusions

Sentiment analysis can be helpful to sort through reviews quickly and help users identify trends on a certain movie or topic. Our machine learning models (Skywalker and R2D2) significantly outperformed the rule-based VADER model, demonstrating the effectiveness of supervised learning approaches for sentiment classification.

### License

This project is for educational purposes as part of a data science course.

### Acknowledgments

- IMDB Dataset: Large Movie Review Dataset
- Rotten Tomatoes Dataset: Cornell Movie Review Data
- VADER: Valence Aware Dictionary for Sentiment Reasoning
