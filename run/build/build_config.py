import sys
sys.path.append('../', 0)
from const import RAW_DATA_PATH, REPO_PATH, PROCESSED_DATA_PATH, FEATURES_PATH, DATA_PROCESSOR_PATH
from fbref_const import TARGET_COLUMNS

ALL_SEASONS = [
    '2017-18',
    '2018-19',
    '2019-20',
    '2020-21',
    '2021-22',
    '2022-23',
    '2023-24',
    '2024-25',
]

CURRENT_SEASON = '2024-25'
assert CURRENT_SEASON in ALL_SEASONS, f"Current season {CURRENT_SEASON} is not in the list of all seasons."

