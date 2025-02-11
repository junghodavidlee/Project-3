 <img src= "https://legacybox.com/cdn/shop/articles/LBFilmReel_991x.progressive.jpg?v=1563402798" alt="Alt Text" width="800" height="350">

# Project-3 Movie Sentiment Analysis
## Jungho Lee, James Kim, Maddie Jankowski

### Project Overview

**Background:**
Movie review sites can have hundreds of thousands of entries with different sentiments. 

**Aim:**
We created a model to provide sentiment analysis for each review automatically. 

### Dataset Overview
We used two datasets, one from IMDB and one from Rotten Tomatoes. The IMDB contained 50k reviews and the RT contained about 8k. 

The target column of both datasets is a positive or negative label of the review.

### Approach to Project goals 
This project compares the performance of our machine learning model with the VADER sentiment classification model. We also used Spacy and BERT to elicit NER from the reviews and see if it had correlation to the positive or negative review. The NER results for PERSON entities and all entities were inconclusive. 

### Results
After fine-tuning our model, the F1 score is 89%. VADER has historically had a 65-70% accuracy on text such as tweets or social media. Because of the casual nature of these reviews, we believe the model performed well on these reviews as well as it would on social media. 

## Conclusions
Sentiment analysis can be helpful to sort through reviews and assess trends on a certain topic. Our model performed well as well as the VADER model.
