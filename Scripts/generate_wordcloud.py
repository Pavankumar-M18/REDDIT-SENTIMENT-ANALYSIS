import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('Data/reddit_sentiment_results.csv')

def generate_wordclouds(df, sentiment_col, text_col, prefix='vader'):
    sentiments = df[sentiment_col].unique()
    for sentiment in sentiments:
        text = ' '.join(df[df[sentiment_col] == sentiment][text_col].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate(text)

        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"{prefix.upper()} word cloud - {sentiment} sentiment")
        plt.tight_layout()
        plt.savefig(f"plots/{prefix}_wordcloud_{sentiment.lower()}.png")
        plt.close()


# Generate wordcloud for both VADER and Textblob results
generate_wordclouds(df, 'VADER_Label', 'Cleaned_Text', prefix='vader')
generate_wordclouds(df, 'TextBlob_Label', 'Cleaned_Text', prefix='textblob')


print("Word clouds generated and saved in 'plots/' folder.")