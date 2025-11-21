from const import RAW_DATA_PATH, MODELS_PATH, PROCESSED_DATA_PATH, FEATURES_PATH, DATA_PROCESSOR_PATH
from fbref_const import TARGET_COLUMNS

SEASONS = [
    '2017-18',
    '2018-19',
    '2019-20',
    '2020-21',
    '2021-22',
    '2022-23',
    '2023-24',
    '2024-25',
]

TARGET_RANGES={
    'home_goals': [0,5],
    'away_goals': [0,6],
    'home_corners': [3,7],
    'away_corners': [3,7],
    'home_cards': [0, 7],
    'away_cards': [0, 7],
    'home_shots': [5, 20],
    'away_shots': [5, 20],
    'home_sots': [1, 10],
    'away_sots':[1, 10],
}

COMPETITION = 'premier_league'

DESTS=[
    f"{RAW_DATA_PATH}/{COMPETITION}/{season}/" for season in SEASONS
]

COMP_PROCESSED_DATA_PATH=f'{PROCESSED_DATA_PATH}/{COMPETITION}/'
COMP_FEATURES_PATH=f'{FEATURES_PATH}/{COMPETITION}/'

INFERENCE_CONFIG = {
    'target_ranges': TARGET_RANGES,
    'competition': COMPETITION,
    'seasons': SEASONS,
    'processed_path': COMP_PROCESSED_DATA_PATH,
    'features_path': COMP_FEATURES_PATH,
    'data_processor_path': f'{DATA_PROCESSOR_PATH}/{COMPETITION}/',
    'save_model_path': f"{MODELS_PATH}/{COMPETITION}/",
}