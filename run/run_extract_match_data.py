from run_config import REPO_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, SEASONS
from fbref_const import TARGET_COLUMNS
import sys
sys.path.insert(1, f"{REPO_PATH}")

from run_config import REPO_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, SEASONS

import pandas as pd
import os

from src.data.match_data_processing import process_match_target_var, process_match_other_var

if __name__=='__main__':
    for season in SEASONS:
        season_path=f'{RAW_DATA_PATH}/{season}/'
        processed_path=f'{PROCESSED_DATA_PATH}/{season}/'
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        all_matches=pd.read_csv(f'{season_path}/all_matches.csv').iloc[:,1].tolist()
        target_df=[]
        for match_name in all_matches:
            home_df=pd.read_csv(f'{season_path}/{match_name}_home_player_stat.csv')
            away_df=pd.read_csv(f'{season_path}/{match_name}_away_player_stat.csv')
            match_df=pd.read_csv(f'{season_path}/{match_name}_match_stat.csv', index_col=0)
            target_df.append(process_match_target_var(home_df, away_df, match_df, match_name))
        target_df=pd.concat(target_df, axis=0)
        target_df.to_csv(f'{processed_path}/all_target_df.csv', index=False)
    
    for season in SEASONS:
        season_path=f'{RAW_DATA_PATH}/{season}/'
        processed_path=f'{PROCESSED_DATA_PATH}/{season}/'
        all_matches=pd.read_csv(f'{season_path}/all_matches.csv').iloc[:,1].tolist()
        data_df=[]
        for match_name in all_matches:
            home_df=pd.read_csv(f'{season_path}/{match_name}_home_player_stat.csv')
            away_df=pd.read_csv(f'{season_path}/{match_name}_away_player_stat.csv')
            match_df=pd.read_csv(f'{season_path}/{match_name}_match_stat.csv', index_col=0)
            data_df.append(process_match_other_var(home_df, away_df, match_df, match_name, TARGET_COLUMNS))
        data_df=pd.concat(data_df, axis=0)
        data_df.to_csv(f'{processed_path}/all_data_df.csv', index=False)