# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set Visualization style
sns.set(style='whitegrid')


# Load the sentiment results
df = pd.read_csv('data/reddit_sentiment_results.csv')


# Sentiment distribution VADER
plt.figure(figsize=(6,4))
sns.countplot(x='VADER_Label', data = df, palette='coolwarm')
plt.title("VADER Sentiment Disgtribution")
plt.xlabel("Sentiment Label")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.savefig('plots/vader_sentiment_distribution.png')
plt.show()


# Sentiment Distribution TextBlob
plt.figure(figsize=(6, 4))
sns.countplot(x='TextBlob_Label', data = df, palette='viridis')
plt.title("TextBlob Sentiment Distribution")
plt.xlabel("Sentiment Label")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.savefig('plots/textblob_sentiment_distribution.png')
plt.show()


# Compare VADER vs TextBlob - Side by Side
comparison = pd.crosstab(df['VADER_Label'], df['TextBlob_Label'])
comparison.plot(kind='bar', stacked=True, figsize=(8,5), colormap='Accent')
plt.title("VADER vs TextBlob Sentiment Comparison")
plt.xlabel("VADER Sentiment")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.savefig('plots/vader_vs_textblob_comparision.png')
plt.show()


# sentiment distribution by subreddit (VADER)
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Subreddit', hue='VADER_Label', palette='Set2')
plt.title("VADER Sentiment by Subreddit")
plt.xlabel("Subreddit")
plt.ylabel("Number of Posts")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig('plots/vader_sentiment_by_subreddit.png')
plt.show()


# Post Score vs Sentiment Score
plt.figure(figsize=(8, 5))
sns.boxplot(x='VADER_Label', y='Score', data=df, palette='Set3')
plt.title("Reddit Post Score vs VADER Sentiment")
plt.xlabel("Sentiment Label")
plt.ylabel("Post Score")
plt.tight_layout()
plt.savefig('plots/post_score_vs_sentiment.png')
plt.show()


print("Visualizations Completed and saved in 'plots/' folder.")