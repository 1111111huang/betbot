from const import PROCESSED_DATA_PATH, FEATURES_PATH
TARGET_RANGES={
    'home_goals': [0,5],
    'away_goals': [0,6],
    #'home_corners': [3,7],
    #'away_corners': [3,7],
    #'home_cards': [0, 7],
    #'away_cards': [0, 7],
    #'home_shots': [5, 20],
    #'away_shots': [5, 20],
    #'home_sots': [1, 10],
    #'away_sots':[1, 10],
}

SEASONS=[
    '2017-18',
    '2018-19',
    '2019-20',
    '2020-21',
    '2021-22',
    '2022-23',
    '2023-24',
    '2024-25',
]

COMPETITION = 'premier_league'
COMP_PROCESSED_DATA_PATH=f'{PROCESSED_DATA_PATH}/{COMPETITION}/'
COMP_FEATURES_PATH=f'{FEATURES_PATH}/{COMPETITION}/'

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

EXPERIMENT_NAME = "mlflow_distribution_ts_5fold_0.5train_2017-24_lag_target"

KEY_COLUMNS=['home', 'away', 'date']

TRAINGING_CONFIG = {
    'seasons': SEASONS,
    'target_ranges': TARGET_RANGES,
    'competition': COMPETITION,
    'processed_data_path': COMP_PROCESSED_DATA_PATH,
    'features_path': COMP_FEATURES_PATH,
    'test_size': TEST_SIZE,
    'validation_size': VALIDATION_SIZE,
    'random_state': RANDOM_STATE,
    'experiment_name': EXPERIMENT_NAME,
    'key_columns': KEY_COLUMNS
}