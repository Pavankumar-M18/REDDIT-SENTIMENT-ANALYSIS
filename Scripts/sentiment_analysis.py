# Importing Libraries

#import nltk
#nltk.download('vader_lexicon')

import pandas as pd
from textblob import TextBlob        # Provides sentiment polarity scores
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Load the Dataset
df = pd.read_csv('cleaned_reddit_data.csv')


# Initialize Vader Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()


# ----------- Textblob Sentiment Analysis ------------
def get_textblob_sentiment(text):
    text = str(text) if isinstance(text, str) else ""      # Converts any NaN, float, or None values to an empty string
    if text.strip():                                    # Checks if the text is not empty
        analysis = TextBlob(text)
        return analysis.sentiment.polarity              # Returns sentiment score between -1 and 1
    return 0                                           # Default neutral sentiment for missing/empty values


# ----------- Vader Sentiment Analysis --------------
def get_vader_sentiment(text):
    text = str(text) if isinstance(text, str) else ""
    if text.strip():
        scores = analyzer.polarity_scores(text)      # Vader returns compound sentiment score
        return scores['compound']
    return 0                           # Default neutral sentiment for missing/empty values


# ----------- Apply Sentiment analysis on Data ---------------
df['TextBlob_Sentiment'] = df['Cleaned_Text'].apply(get_textblob_sentiment)   # Uses textblob to get sentiment score
df['VADER_Sentiment'] = df['Cleaned_Text'].apply(get_vader_sentiment)      # Uses VADER to get sentiment score


# -------------- Classify Sentiment as Positive, Neutral or Negative ---------------

def classify_sentiment(score):          # Based on the sentiment score function assaigns sentiment score +ve, -ve, Neutral.
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"


# Converts numerical sentiment scores into labels for better understanding.

df['TextBlob_Label'] = df['TextBlob_Sentiment'].apply(classify_sentiment)
df['VADER_Label'] = df['VADER_Sentiment'].apply(classify_sentiment)


# Save the Result

df.to_csv('reddit_sentiment_results.csv', index=False)

print("Sentiment Analysis Completed Successfully!")