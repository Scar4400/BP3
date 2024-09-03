import os
from dotenv import load_dotenv

load_dotenv()

API_KEYS = {
    "pinnacle": os.getenv("PINNACLE_API_KEY"),
    "livescore": os.getenv("LIVESCORE_API_KEY"),
    "api_football": os.getenv("API_FOOTBALL_KEY")
}

URLS = {
    "pinnacle": "<https://api.pinnacle.com/v1/odds>",
    "livescore": "<https://api.livescore.com/v1/api/app/date>",
    "api_football": "<https://api-football-v1.p.rapidapi.com/v3/odds>"
}

HEADERS = {
    "pinnacle": {"Authorization": f"Bearer {API_KEYS['pinnacle']}"},
    "livescore": {"X-RapidAPI-Key": API_KEYS['livescore']},
    "api_football": {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": API_KEYS['api_football']
    }
}

