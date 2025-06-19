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

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

EXPERIMENT_NAME = "mlflow_distribution_ts_kfold_0.5train_test"

KEY_COLUMNS=['home', 'away', 'date']