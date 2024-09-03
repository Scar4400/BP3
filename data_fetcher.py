import requests
import pandas as pd
from config import URLS, HEADERS
from pytrends.request import TrendReq
from bs4 import BeautifulSoup
from textblob import TextBlob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data(url, headers, params=None):
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None

def fetch_api_data():
    data = {}
    for source, url in URLS.items():
        data[source] = fetch_data(url, HEADERS[source])
    return data

def fetch_google_trends(teams):
    pytrends = TrendReq(hl='en-US', tz=360)
    trends_data = {}
    for team in teams:
        pytrends.build_payload([team], cat=0, timeframe='now 7-d')
        trends_data[team] = pytrends.interest_over_time().mean()[team]
    return trends_data

def fetch_news_sentiment(teams):
    sentiment_data = {}
    for team in teams:
        url = f"<https://news.google.com/rss/search?q={team}+football&hl=en-US&gl=US&ceid=US:en>"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, features="xml")
        titles = soup.findAll('title')[1:6]  # Get top 5 news titles
        sentiment = sum([TextBlob(title.text).sentiment.polarity for title in titles]) / len(titles)
        sentiment_data[team] = sentiment
    return sentiment_data

def fetch_all_data():
    api_data = fetch_api_data()

    # Extract team names (this is a placeholder, adjust based on actual data structure)
    teams = set()
    for source, data in api_data.items():
        if data and 'teams' in data:
            teams.update(data['teams'])

    trends_data = fetch_google_trends(teams)
    sentiment_data = fetch_news_sentiment(teams)

    return {**api_data, "trends": trends_data, "sentiment": sentiment_data}

if __name__ == "__main__":
    data = fetch_all_data()
    for source, content in data.items():
        if content:
            logger.info(f"{source.capitalize()} data fetched successfully")
        else:
            logger.warning(f"Failed to fetch {source} data")

