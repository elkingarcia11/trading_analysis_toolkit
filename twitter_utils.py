import tweepy
import os

def tweet(message):
    """Post a tweet with the given message."""
    try:
        client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_KEY_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
            wait_on_rate_limit=True
        )
        client.create_tweet(text=message)
    except tweepy.TweepyException as e:
        print("Tweepy Exception:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)
