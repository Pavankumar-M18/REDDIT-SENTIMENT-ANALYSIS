import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


# Load the cleaned dataset
df = pd.read_csv('Data/reddit_sentiment_results.csv')


# Drop Nan values or missing values
df.dropna(subset=['Cleaned_Text', 'VADER_Label'], inplace=True)

# Convert Sentiment Labels into Numeric values
sentiment_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
df['Sentiment_Label'] = df['VADER_Label'].map(sentiment_mapping)


# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

x = tfidf_vectorizer.fit_transform(df['Cleaned_Text'])
y = df['Sentiment_Label']


# split data into Train and Test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# Apply SMOTE to balance classes in training data
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# Initialize models
logistic_model = LogisticRegression()
svm_model = SVC(kernel='linear')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
logistic_model.fit(x_train_smote, y_train_smote)
svm_model.fit(x_train_smote, y_train_smote)
rf_model.fit(x_train_smote, y_train_smote)


# Predictions
y_pred_logistic = logistic_model.predict(x_test)
y_pred_svm = svm_model.predict(x_test)
y_pred_rf = rf_model.predict(x_test)


# Evaluate models
print("Logistic Regressions Performance after SMOTE:")
print(classification_report(y_test, y_pred_logistic))

print("Support Vector Machine Performance after SMOTE:")
print(classification_report(y_test, y_pred_svm))

print("Random Forest Performance after SMOTE:")
print(classification_report(y_test, y_pred_rf))