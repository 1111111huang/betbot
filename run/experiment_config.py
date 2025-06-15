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

TEST_SIZE = 0.2
RANDOM_STATE = 42

EXPERIMENT_NAME = "mlflow_distribution_prediction_test"

MODEL_PARAMS={
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_STATE
}

KEY_COLUMNS=['home', 'away', 'date']