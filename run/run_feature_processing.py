from run_config import REPO_PATH, DATA_PROCESSOR_PATH, PROCESSED_DATA_PATH, SEASONS
from fbref_const import TARGET_COLUMNS
import sys
sys.path.insert(1, f"{REPO_PATH}")

import pandas as pd
from src.feature.feature_encoders import TeamEncoder, TeamLagFeatureGenerator, PreviousSeasonTeamAverager

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
    team_encoding_df = encoder.transform(data_dfs)

    # Previous Season Team Average
    previous_season_feature= PreviousSeasonTeamAverager(decay_factor=1, date_col='date', home_col='home', away_col='away')
    prev_season_feature_df = previous_season_feature.transform(data_dfs)

    # Team Lag Features
    generator = TeamLagFeatureGenerator(lookback=5)
    team_lag_feature_df = generator.transform(data_dfs)

    feaature_df= pd.concat([team_encoding_df, prev_season_feature_df, team_lag_feature_df], axis=1)
