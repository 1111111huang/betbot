from data_etl_config import SCRAPER_CONFIG
from const import REPO_PATH
import sys
sys.path.insert(1, f"{REPO_PATH}")

from fbref_const import URLs

from src.data.scraper import scrape_season_match_data

if __name__=='__main__':
    for season, dest in zip(SCRAPER_CONFIG['seasons'], SCRAPER_CONFIG['dests']):
        url = URLs[SCRAPER_CONFIG['competition']][season]
        print(f"Scraping {url} to {dest}")
        # Run the scraper
        scrape_season_match_data(url, dest)