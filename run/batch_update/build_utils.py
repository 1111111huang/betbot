from const import REPO_PATH
import sys
sys.path.insert(0, f"{REPO_PATH}")
from src.data.scraper import scrape_season_match_data
from src.data.match_data_processing import process_match_target_var, process_match_other_var

import pandas as pd
import os

def get_dests(raw_data_paths, competition, seasons):
    """
    Get the destination paths for each season.
    """
    return [f"{raw_data_paths}/{competition}/{season}" for season in seasons]

def get_processed_path(processed_data_path, competition):
    """
    Get the processed data path for a given competition.
    """
    return f"{processed_data_path}/{competition}"

def scrape_competitions_for_seasons(all_seasons, raw_data_path, competitions, urls):
    for competition in competitions:
        for season, dest in zip(all_seasons, get_dests(raw_data_path, competition, all_seasons)):
            url = urls[competition][season]
            print(f"Scraping {url} to {dest}")
            # Run the scraper
            scrape_season_match_data(url, dest)

def check_if_target_df_exists(processed_path, all_matches_df):
    """
    Check if the target DataFrame already exists and matches the expected length.
    """
    target_path = f"{processed_path}/all_target_df.csv"
    if os.path.exists(target_path):
        df = pd.read_csv(target_path)
        return len(df) == len(all_matches_df)
    return False

def check_if_data_df_exists(processed_path, all_matches_df):
    """
    Check if the data DataFrame already exists and matches the expected length.
    """
    data_path = f"{processed_path}/all_data_df.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return len(df) == len(all_matches_df)
    return False

def process_raw_match_data(all_competitions, all_seasons, raw_data_path, processed_data_path, target_columns, verbose=True):
    for competition in all_competitions:
        for season in all_seasons:
            season_path = f'{raw_data_path}/{competition}/{season}/'
            processed_path = f'{get_processed_path(processed_data_path, competition)}/{season}/'
            if not os.path.exists(processed_path):
                os.makedirs(processed_path)
            all_matches = pd.read_csv(f'{season_path}/all_matches.csv').iloc[:, 1].tolist()
            # Check if target df exists and matches length
            if check_if_target_df_exists(processed_path, all_matches):
                if verbose:
                    print(f"Target DataFrame for {competition} {season} already exists and matches. Skipping.")
            else:
                if verbose:
                    print(f"Processing target DataFrame for {competition} {season}.")
                target_df = []
                for match_name in all_matches:
                    home_df = pd.read_csv(f'{season_path}/{match_name}_home_player_stat.csv')
                    away_df = pd.read_csv(f'{season_path}/{match_name}_away_player_stat.csv')
                    match_df = pd.read_csv(f'{season_path}/{match_name}_match_stat.csv', index_col=0)
                    target_df.append(process_match_target_var(home_df, away_df, match_df, match_name))
                target_df = pd.concat(target_df, axis=0)
                target_df.to_csv(f'{processed_path}/all_target_df.csv', index=False)

        for season in all_seasons:
            season_path = f'{raw_data_path}/{competition}/{season}/'
            processed_path = f'{get_processed_path(processed_data_path, competition)}/{season}/'
            all_matches = pd.read_csv(f'{season_path}/all_matches.csv').iloc[:, 1].tolist()
            # Check if data df exists and matches length
            if check_if_data_df_exists(processed_path, all_matches):
                if verbose:
                    print(f"Data DataFrame for {competition} {season} already exists and matches. Skipping.")
            else:
                if verbose:
                    print(f"Processing data DataFrame for {competition} {season}.")
                data_df = []
                for match_name in all_matches:
                    home_df = pd.read_csv(f'{season_path}/{match_name}_home_player_stat.csv')
                    away_df = pd.read_csv(f'{season_path}/{match_name}_away_player_stat.csv')
                    match_df = pd.read_csv(f'{season_path}/{match_name}_match_stat.csv', index_col=0)
                    data_df.append(process_match_other_var(home_df, away_df, match_df, match_name, target_columns))
                data_df = pd.concat(data_df, axis=0)
                data_df.to_csv(f'{processed_path}/all_data_df.csv', index=False)

