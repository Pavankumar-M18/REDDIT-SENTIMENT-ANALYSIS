# Importing Libraries

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download NLTK Resoources

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# Load the Dataset
df = pd.read_csv('reddit_data.csv')

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Drop Unneccessary Columns
if 'URL' in df.columns:
    df.drop(columns=['URL'], inplace=True)

# Fill missing text with empty string
df.loc[:, 'Text'] = df['Text'].fillna("")

# Initialize NLP Tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()      # Convert to Lower case
    text = re.sub(r'http\S+|www\S+', '', text)     # Remove URLS
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # Remove special Charecters & punctuations

    words = word_tokenize(text)      # Tokenization

    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]    # Remove stopwords & apply Lemmatization
    return " ".join(words) if words else "empty_text"


# Apply text cleaning to the Text Column
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Drop rows where "Cleaned_text" is "empty_text"
df = df[df['Cleaned_Text'] != "empty_text"]

# save the Cleaned Data
df.to_csv('cleaned_reddit_data.csv', index=False)

print(f"✅ Data Preprocessing completed successfully! Rows after cleaning: {len(df)}")