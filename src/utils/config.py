import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

API_FOOTBALL_BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
WEATHER_API_BASE_URL = "http://api.weatherapi.com/v1"

# Data Configuration
LEAGUE_IDS = {
    'premier_league': 39
}
SEASONS = ['2020', '2021', '2022', '2023']
RATE_LIMIT = 300  # requests per minute

# File Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

# CSV Files
INPUT_DATA_FILE = os.path.join(RAW_DATA_DIR, 'input_data.csv')
LATEST_DATA_FILE = os.path.join(RAW_DATA_DIR, 'latest.csv')
PREMIER_LEAGUE_2024_FILE = os.path.join(RAW_DATA_DIR, 'PremierLeagueSeason2024.csv')
PLAYER_STATS_FILE = os.path.join(RAW_DATA_DIR, '2022-2023_Football_Player_Stats.csv')
PREMIER_22_23_FILE = os.path.join(RAW_DATA_DIR, 'Premier22_23.csv')
PREMIER_21_22_FILE = os.path.join(RAW_DATA_DIR, 'Premier21-22.csv')
UPCOMING_FIXTURES_FILE = os.path.join(RAW_DATA_DIR, 'upcoming_fixtures.csv')

# Model Configuration
MODEL_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Feature Engineering Configuration
ROLLING_WINDOWS = [3, 5, 10]  # Last n matches
EMA_SPANS = [3, 5, 10]  # Exponential moving averages

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)
