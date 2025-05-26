from run_config import REPO_PATH, SEASONS, dests
import sys
sys.path.insert(1, f"{REPO_PATH}")

from fbref_const import URLs

from src.data.scraper import scrape_season_match_data

if __name__=='__main__':
    for season, dest in zip(SEASONS, dests):
        url = URLs[season]
        print(f"Scraping {url} to {dest}")
        # Run the scraper
        scrape_season_match_data(url, dest)