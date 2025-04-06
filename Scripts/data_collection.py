# Importing Libraries

import praw        # python Reddit API Wraper
import os
from dotenv import load_dotenv      # To load API credentials from .env file
import pandas as pd


# Load Environment Variables from .env file
load_dotenv()


# ------------------ Access Reddit API Credentials --------------------
 
# Retrives the CLIENT_ID,CLIENT_SECRET,USER_AGENT from the environment variables.
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USER_AGENT = os.getenv('USER_AGENT')


# ------------ Authenticate with Reddit API -------------

# Establish a connection to Reddit using Oath Authentication.
reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

# ----------- Select the Subreddit ------------------

subreddits = ("StockMarket", "investing", "stocks", "finance")
categories = ["hot", "new", "top", "rising"]


# ------------- Collect Posts ----------------

posts = []        # List to hold the Posts data


# Fetches posts from each subreddit and category

for sub in subreddits:
    subreddit = reddit.subreddit(sub)


    for category in categories:
        if category == "hot":
            submissions = subreddit.hot(limit=4000)
        elif category == "new":
            submissions = subreddit.new(limit=4000)
        elif category == "top":
            submissions = subreddit.top(limit=4000)
        elif category == "rising":
            submissions = subreddit.rising(limit=4000)
        
        count = 0
        for submission in submissions:
            posts.append([sub, submission.title, submission.selftext, submission.score, submission.upvote_ratio, submission.url])
            count += 1
        print(f"Fetched {count} posts from r/{sub} ({category})")



# ------------------ Save the data -----------------

# Converts the collected data into a Pandas DataFrame

df = pd.DataFrame(posts, columns=['Subreddit', 'Title', 'Text', 'Score', 'Upvote Ratio', 'URL'])

print(f"Collected {len(posts)} posts in total.")

df.to_csv('reddit_data.csv', index=False)       # Saves the data to csv file


print(f"Data Collection completed Successfully! Total posts collected: {len(df)}")