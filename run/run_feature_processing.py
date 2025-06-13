from run_config import REPO_PATH, DATA_PROCESSOR_PATH, PROCESSED_DATA_PATH, SEASONS, FEATURES_PATH
from fbref_const import TARGET_COLUMNS
import sys
sys.path.insert(1, f"{REPO_PATH}")

import pandas as pd
from src.feature.feature_encoders import TeamEncoder, TeamLagFeatureGenerator, PreviousSeasonTeamAverager, TeamRestDaysCalculator

if __name__ == '__main__':
    seasons=sorted(SEASONS)
    data_dfs=[pd.read_csv(f"{PROCESSED_DATA_PATH}/{season}/all_data_df.csv") for season in seasons]
    print(len(data_dfs), 'seasons loaded')

    key_columns = ['home', 'away', 'date']

    # Team Encoding
    encoder = TeamEncoder(n_first_matches=5)
    encoder.fit(data_dfs)  # season_dfs is a list of DataFrames, one per season
    encoder.save(f"{DATA_PROCESSOR_PATH}/team_encoder.pkl")

    encoder = TeamEncoder.load(f"{DATA_PROCESSOR_PATH}/team_encoder.pkl")
    team_encoding_df = encoder.transform(data_dfs).reset_index(drop=True)

    # Previous Season Team Average
    previous_season_feature= PreviousSeasonTeamAverager(decay_factor=1, date_col='date', home_col='home', away_col='away')
    prev_season_feature_df = previous_season_feature.transform(data_dfs).reset_index(drop=True)

    # Team Lag Features
    generator = TeamLagFeatureGenerator(lookback=5)
    team_lag_feature_df = generator.transform(data_dfs[1:]).reset_index(drop=True)

    # Team Rest Days
    team_rest_days_calculator = TeamRestDaysCalculator()
    team_rest_days_features = team_rest_days_calculator.transform(data_dfs[1:]).reset_index(drop=True)

    print('Team rest days features shape:', team_rest_days_features.shape)
    print('Team lag features shape:', team_lag_feature_df.shape)
    print('Team encoding features shape:', team_encoding_df.shape)
    print('Previous season features shape:', prev_season_feature_df.shape)

    # Combine all features
    combined_features = team_encoding_df.merge(prev_season_feature_df, on=key_columns, how='inner')
    combined_features = combined_features.merge(team_lag_feature_df, on=key_columns, how='inner')
    combined_features = combined_features.merge(team_rest_days_features, on=key_columns, how='inner')
    print('Combined features shape:', combined_features.shape)
    combined_features.to_csv(f"{FEATURES_PATH}/all_combined_features.csv", index=False)

