from const import REPO_PATH
from data_etl_config import FEATURE_CONFIG
import sys
sys.path.insert(1, f"{REPO_PATH}")
import os

import pandas as pd
from src.feature.feature_encoders import TeamEncoder, TeamLagFeatureGenerator, PreviousSeasonTeamAverager, TeamRestDaysCalculator, TeamLagTargetFeature
from sklearn.preprocessing import StandardScaler
import joblib

if __name__ == '__main__':
    seasons=sorted(FEATURE_CONFIG['seasons'])
    data_dfs=[pd.read_csv(f"{FEATURE_CONFIG['processed_path']}/{season}/all_data_df.csv") for season in seasons]
    target_dfs=[pd.read_csv(f"{FEATURE_CONFIG['processed_path']}/{season}/all_target_df.csv") for season in seasons[1:]]
    print(len(data_dfs), 'seasons loaded')

    key_columns = ['home', 'away', 'date']

    # Team Encoding
    encoder = TeamEncoder(n_first_matches=5)
    encoder.fit(data_dfs)  # season_dfs is a list of DataFrames, one per season
    encoder.save(f"{FEATURE_CONFIG['data_processor_path']}/team_encoder.pkl")

    encoder = TeamEncoder.load(f"{FEATURE_CONFIG['data_processor_path']}/team_encoder.pkl")
    team_encoding_df = encoder.transform(data_dfs).reset_index(drop=True)

    # Team Lag Features
    generator = TeamLagFeatureGenerator(lookback=5)
    team_lag_feature_df = generator.transform(data_dfs[1:]).reset_index(drop=True)

    # Team Rest Days
    team_rest_days_calculator = TeamRestDaysCalculator()
    team_rest_days_features = team_rest_days_calculator.transform(data_dfs[1:]).reset_index(drop=True)

    lag_target_feature = TeamLagTargetFeature()
    team_lag_target_df = lag_target_feature.transform(data_dfs[1:], target_dfs).reset_index(drop=True)


    print('Team rest days features shape:', team_rest_days_features.shape)
    print('Team lag features shape:', team_lag_feature_df.shape)
    print('Team encoding features shape:', team_encoding_df.shape)
    print('Team lag target features shape:', team_lag_target_df.shape)

    # Combine all features
    combined_features = team_encoding_df.copy()
    combined_features = combined_features.merge(team_lag_feature_df, on=key_columns, how='inner')
    combined_features = combined_features.merge(team_rest_days_features, on=key_columns, how='inner')
    combined_features = combined_features.merge(team_lag_target_df, on=key_columns, how='inner')
    print('Combined features shape:', combined_features.shape)
    if not os.path.exists(FEATURE_CONFIG['features_path']):
            os.makedirs(FEATURE_CONFIG['features_path'])
    # Identify one-hot encoded columns (from TeamEncoder)
    onehot_prefixes = ['encoded_home', 'encoded_away']
    numerical_cols = [col for col in combined_features.columns if (combined_features[col].dtype in ['float64', 'int64']) and not any(col.startswith(prefix) for prefix in onehot_prefixes)]
    scaler = StandardScaler()
    combined_features[numerical_cols] = scaler.fit_transform(combined_features[numerical_cols])
    # Save scaler to the same folder as the encoder
    encoder_folder = FEATURE_CONFIG['data_processor_path']
    scaler_path = os.path.join(encoder_folder, 'standard_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler saved to {scaler_path}")
    combined_features.to_csv(f"{FEATURE_CONFIG['features_path']}/all_combined_features_fixed_lag_2017-24.csv", index=False)

