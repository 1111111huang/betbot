from const import RAW_DATA_PATH, REPO_PATH, PROCESSED_DATA_PATH, FEATURES_PATH, DATA_PROCESSOR_PATH
from fbref_const import TARGET_COLUMNS

SEASONS = [
    '2025-26'
]

COMPETITION = 'premier_league'

DESTS=[
    f"{RAW_DATA_PATH}/{COMPETITION}/{season}/" for season in SEASONS
]

COMP_PROCESSED_DATA_PATH=f'{PROCESSED_DATA_PATH}/{COMPETITION}/'
COMP_FEATURES_PATH=f'{FEATURES_PATH}/{COMPETITION}/'

SCRAPER_CONFIG = {
    'competition': COMPETITION,
    'seasons': SEASONS,
    'dests': DESTS
}

PROCESSED_DATA_CONFIG = {
    'competition': COMPETITION,
    'target_columns': TARGET_COLUMNS,
    'seasons': SEASONS,
    'processed_path': COMP_PROCESSED_DATA_PATH,
}

FEATURE_CONFIG = {
    'competition': COMPETITION,
    'seasons': SEASONS,
    'processed_path': COMP_PROCESSED_DATA_PATH,
    'features_path': COMP_FEATURES_PATH,
    'data_processor_path': f'{DATA_PROCESSOR_PATH}/{COMPETITION}/',
}