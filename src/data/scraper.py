from wsgiref import headers
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import cloudscraper

from src.data.match_data_processing import *

def get_season_match_links(url):
    scraper = cloudscraper.create_scraper()
    data = scraper.get(url)
    if data.status_code != 200:
        raise Exception(f"Failed to fetch page: {url} (status code: {data.status_code})")
    soup = BeautifulSoup(data.text, 'html.parser')
    links = soup.select('table.stats_table')[0].find_all('a')
    links = [l.get("href") for l in links]
    links = list(set([l for l in links if '/matches/' in l and len(l)>22]))
    return links

def scrape_season_match_data(url, dest, verbose=True):
    if not os.path.exists(dest):
        os.makedirs(dest)
    links = get_season_match_links(url)
    all_matches_df = pd.DataFrame(list(map(lambda l: l.split('/')[-1], links)))

    csv_path = f'{dest}/all_matches.csv'
    # Check if all_matches.csv exists and if it's the same as the current dataframe
    already_scraped_flag = False
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, index_col=0)
        if all_matches_df.equals(existing_df):
            already_scraped_flag = True
            if verbose:
                print("all_matches.csv already exists and is identical. Skipping save.")
        else:
            if verbose:
                print("all_matches.csv exists but is different. Overwriting.")
            all_matches_df.to_csv(csv_path)
    else:
        all_matches_df.to_csv(csv_path)

    for link in links:
        # Check if all 4 files for this match already exist
        match_id = link.split("/")[-1]
        match_stat_path = f'{dest}/{match_id}_match_stat.csv'
        home_player_stat_path = f'{dest}/{match_id}_home_player_stat.csv'
        away_player_stat_path = f'{dest}/{match_id}_away_player_stat.csv'
        shot_stat_path = f'{dest}/{match_id}_shot_stat.csv'
        if all(os.path.exists(p) for p in [match_stat_path, home_player_stat_path, away_player_stat_path, shot_stat_path]):
            if verbose:
                print(f"Files for match {match_id} already exist. Skipping.")
            continue

        dfs = pd.read_html(f"https://fbref.com{link}")
        results = process_match_data(dfs)

        results[0].to_csv(match_stat_path)
        results[1].to_csv(home_player_stat_path)
        results[2].to_csv(away_player_stat_path)
        results[3].to_csv(shot_stat_path)
        time.sleep(20)