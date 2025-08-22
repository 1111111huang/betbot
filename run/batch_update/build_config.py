import sys
sys.path.insert(0, '../')
sys.path.insert(1, '../../')

ALL_SEASONS = [
    '2017-18',
    '2018-19',
    '2019-20',
    '2020-21',
    '2021-22',
    '2022-23',
    '2023-24',
    '2024-25',
    '2025-26',
]

CURRENT_SEASON = '2025-26'
assert CURRENT_SEASON in ALL_SEASONS, f"Current season {CURRENT_SEASON} is not in the list of all seasons."

COMPETITIONS = ('premier_league',)# 'la_liga')

INPUT_CSV_PATHS = {
    'premier_league': 'data/inputs/2025-26-01.csv',
}

OUTPUT_CSV_PATH = 'data/outputs/'

TARGET_RANGES={
    'home_goals': [0,5],
    'away_goals': [0,6],
}