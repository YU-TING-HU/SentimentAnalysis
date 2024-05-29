# -*- coding: utf-8 -*-
"""SentimentAnalysis.ipynb
# Data - Customer Reviews of Hotels
"""

import pandas as pd
from google.colab import files

# Upload and read hotel reviews data
uploaded = files.upload()
data = pd.read_csv("hotel_reviews.csv")

# Display basic information about the dataset
data.shape
data.head()
data.tail()
data.info()

print(f"\nData columns:\n{data.columns}") # column names
print(f"\nDescribe:\n{data.describe()}") # statistical summary of the dataset
print(f"\nTypes:\n{data.dtypes}") # data types of each column


"""## Cleaning Data"""

# Check and print rows with missing values
print("Rows with Missing Values:")
data[data.isnull().any(axis=1)]

# Check and print duplicate rows
print("Duplicate Rows:")
data[data.duplicated()]

# Drop rows with missing values and print the new shape
data.dropna(axis=0,inplace=True)
data.shape # (7001, 7) -> (6994, 7)
# Drop duplicate rows and print the new shape
data.drop_duplicates(inplace=True)
data.shape


"""## Correlation Coefficient between Rating & Sentiment  - Using TextBlob"""

from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

# Function to analyze sentiment polarity using TextBlob
def sentiment_analysis(text):
  """Analyze sentiment polarity of a given text."""
  analysis = TextBlob(str(text))
  return analysis.sentiment.polarity

# Apply sentiment analysis on the review texts
data["Sentiment"] = data["Review_Text"].apply(sentiment_analysis)
print("DataFrame with Sentiment Column:")
data

# Calculate Pearson correlation coefficient between rating and sentiment
# Pearson Correlation，相關係數0.3以下為低相關，0.3~0.7為中等相關，0.7以上為高度相關(黃姵嫙，2018)
correlation_coefficient = data["Rating(Out of 10)"].corr(data["Sentiment"])
print(f"\nCorrelation Coefficient between Rating & Sentiment {correlation_coefficient}\n")

# Plot the correlation between ratings and sentiment
plt.figure(figsize=(10,6))
colors = data["Sentiment"].apply(lambda x: 'red' if x < 0 else 'gray' if x == 0 else 'green')
plt.scatter(data["Rating(Out of 10)"], data["Sentiment"], c=colors, alpha=0.7)
plt.title("Correlation Between Ratings and Sentiment")
plt.xlabel("Rating")
plt.ylabel("Sentiment Polarity")
plt.show()


"""## Keyword Analysis of positive & negative reviews - Using TextBlob & NLTK"""

from nltk.corpus import stopwords
from nltk import pos_tag
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Function to categorize sentiment as Positive, Negative, or Neutral
def analyze_sentiment(text):
  analysis = TextBlob(str(text))
  return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

# Apply sentiment analysis on the review texts
data["Sentiment"] = data["Review_Text"].apply(analyze_sentiment)
data

# Concatenate all positive and negative reviews into single strings
positive_reviews = ''.join(data[data["Sentiment"] == "Positive"]["Review_Text"])
negative_reviews = ''.join(data[data["Sentiment"] == "Negative"]["Review_Text"])

# Download NLTK stopwords and tokenize the reviews
stopwords_set = set(stopwords.words("english"))
positive_tokens = word_tokenize(positive_reviews)
negative_tokens = word_tokenize(negative_reviews)

# Remove stopwords and non-alphabetic tokens
positive_tokens = [word.lower() for word in positive_tokens if word.isalpha() and word.lower() not in stopwords_set]
negative_tokens = [word.lower() for word in negative_tokens if word.isalpha() and word.lower() not in stopwords_set]

# Find the top 5 most common words in positive and negative reviews
top_positive_keyowrds = Counter(positive_tokens).most_common(5)
top_negative_keyowrds = Counter(negative_tokens).most_common(5)

print("Top 5 keyowrds that are frequently used in positive reviews")
print(top_positive_keyowrds)
print("\nTop 5 keyowrds that are frequently used in negative reviews")
print(top_negative_keyowrds)


"""## Analyzing the Emotional Aspect of Customer Review - Using EmoLex"""

!pip install NRCLex
from nrclex import NRCLex

# nltk.download("punkt")
data["Emotion"] = data["Review_Text"].apply(lambda text: [(emotion, score * 100) for emotion, score in NRCLex(text).affect_frequencies.items()])

pd.set_option("display.max_colwidth", None)
pd.options.display.float_format = '{:.2%}'.format
data[["Review_Text", "Emotion"]]

# Extract the top emotions from text using EmoLex
# Apply top emotions extraction on the review texts
data["top_emotions"] = data["Review_Text"].apply(lambda text: [emotion for emotion, score in NRCLex(text).top_emotions])
data[["Review_Text", "top_emotions"]]


"""## Sentiment Analysis of Hotel Reviews - Using Textblob"""

# Analyze the overall sentiment of reviews for each hotel
hotel_sentiments = list()
for hotel in data["Name"].unique():
  hotel_df = data[data["Name"] == hotel]

  sentiments = list()
  for review_text in hotel_df["Review_Text"]:
    analysis = TextBlob(str(review_text))
    if analysis.sentiment.polarity > 0:
      sentiments.append("positive")
    elif analysis.sentiment.polarity < 0:
      sentiments.append("negative")
    else:
      sentiments.append("neutral")

  if sentiments:
    overall_sentiment = max(set(sentiments), key=sentiments.count)
  else:
    overall_sentiment = "neutral"

  hotel_sentiments.append({'Hotel':hotel, 'Sentiment':overall_sentiment})

# Convert the list of sentiments to a DataFrame
hotel_sentiments_df = pd.concat([pd.DataFrame(item, index=[0]) for item in hotel_sentiments], ignore_index=True)
hotel_sentiments_df



"""# Data - Twitter Posts

## Analyzing the Emotional Aspect of Twitter Posts - Using EmoLex
"""

import pandas as pd
from google.colab import files
from nrclex import NRCLex
import nltk

# Upload and read Twitter posts data
uploaded = files.upload()
data = pd.read_csv("Tweets.csv")
data

# Cleaning Data
print(f"\nRows with Missing Values:\n{data[data.isnull().any(axis=1)]}\n")
print(f"\nDuplicate Rows:\n{data[data.duplicated()]}\n")
data.dropna(axis=0, inplace=True)
print(f"\nFinal data shape:{data.shape}") # (27481, 4) -> (27480, 4)
data

# Function to analyze emotions in Twitter posts using EmoLex
def emotionanalysis(text):
  top_emotions = [emotion for emotion, score in NRCLex(str(text)).top_emotions]
  return top_emotions

# Apply emotion analysis on the Twitter posts
data["top_emotions"] = data["text"].apply(emotionanalysis)

pd.set_option("display.max_colwidth",None)
pd.options.display.float_format = '{:.2%}'.format

# Print the Twitter texts with their corresponding top emotions
data[["text","top_emotions"]]


"""## Sentiment Analysis of Twitter Posts - Using VADER"""

!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to analyze sentiment using VADER
def sentiment_vader(text):
  analyzer = SentimentIntensityAnalyzer()
  sentiment_scores = analyzer.polarity_scores(str(text))
  if sentiment_scores["compound"] >= 0.05:
    return "Positive"
  elif sentiment_scores["compound"] <= -0.05:
    return "Negative"
  else:
    return "Neutral"

# Apply VADER sentiment analysis on the Twitter posts
data["sentiment_vader"] = data["text"].apply(sentiment_vader)
data[["text","sentiment_vader"]]


"""## Predicting Tweet Sentiment - Using BERT"""

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Function to predict sentiment using BERT
def sentimentanalysis(text):
  inputs = tokenizer(str(text), return_tensors="pt")
  outputs = model(**inputs)
  logits = outputs.logits
  predicted_label = torch.argmax(softmax(logits, dim=1)).item()
  sentiment_mapping = {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}
  predicted_sentiment = sentiment_mapping.get(predicted_label, 'unknown')
  return predicted_sentiment

# Function to interactively predict the sentiment of user input
def sentimentprediction():
  user_input = input("Enter your tweet:")
  sentimentprediction = sentimentanalysis(user_input)
  print(f"Predicted sentiment: {sentimentprediction}")

# Predict sentiment of user input
sentimentprediction()


"""## Predicting Tweet Sentiment - Using MultinomialNB"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Split the data into training and testin
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and MultinomialNB
model = make_pipeline(CountVectorizer(), MultinomialNB())
# Fit the model on the training data
model.fit(train_data["text"].astype(str), train_data["sentiment"])

# Function to predict sentiment using the trained model
def sentimentprediction(tweet):
  predicted_sentiment = model.predict([tweet])[0]
  return predicted_sentiment

# Predict sentiment of user input
user_input = input("Enter your tweet:")
predictedsentiment = sentimentprediction(user_input)
print(f"Predicted sentiment: {predictedsentiment}")

# Predict sentiment of another user input
user_input = input("Enter your tweet:")
predictedsentiment = sentimentprediction(user_input)
print(f"Predicted sentiment: {predictedsentiment}")