 <img src= "https://legacybox.com/cdn/shop/articles/LBFilmReel_991x.progressive.jpg?v=1563402798" alt="Alt Text" width="800" height="350">

# Project-3 Movie Sentiment Analysis
## Jungho Lee, James Kim, Maddie Jankowski

### Project Overview

**Background:**
Movie review sites can have hundreds of thousands of entries with different sentiments. 

**Aim:**
We created 2 models to automatically provide sentiment analysis for each review, targeting users who want to get metadata about a set of reviews of a movie.

### Dataset Overview
We used two datasets, one from IMDB and one from Rotten Tomatoes. The IMDB contained 50k reviews and the RT contained about 8k. 

The target column of both datasets is a positive or negative label of the review.

### Approach to Project goals 
This project compares the performance of our machine learning models with the VADER sentiment classification model. 

Attempted approaches: We also used Spacy and BERT to elicit NER from the reviews and see if the number of entities suggested a more positive or negative review. The NER results for PERSON entities and all entities were inconclusive. 

### Results 
Skywalker (Linear SVC with TDIDF Vectorizer) 92% accuracy on the IMDB dataset
R2D2 (RNN with Tokenizer) 89% accuracy on the IMDB dataset
VADER 70% accuracy on the IMDB dataset

Although the Skywalker had the best-fit model for the IMDB dataset, after evaluating it on the RT dataset, it only produced a 75% accuracy, which still performed better than the VADER. The R2D2 also had a 73% on the RT dataset, while the VADER performed at a low 61% on the dataset. In future investigation, more efforts would be done to look into the smaller, but challenging RT dataset to investigate why our models performed worse on these reviews. 

###  Usage

*Instructions*
1. Clone the repository 
   git clone https://github.com/junghodavidlee/Project-3.git
   
2. Install the following using !pip
!pip install fastparquet \
!pip install gradio \
!pip install vaderSentiment \
!pip install nltk \
!pip install datasets 

4. Run the code using Python script
  
5. Gradio interface can be accessed through the provided link until February 15 or by running the code:
  VADER https://d7cf757c35838543c0.gradio.live/
  SKYWALKER https://3829bd353b5891b653.gradio.live/
  R2D2 https://7fd21291d8f7835edc.gradio.live/

### Conclusions
Sentiment analysis can be helpful to sort through reviews quickly and help users trends on a certain movie or topic. Our models performed better than the VADER model.
