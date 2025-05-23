from scraper_config import urls, dests
import sys
sys.insert(0, '../src/data/')
from scraper import scrape_season_match_data

if __name__=='__main__':
    for url, dest in zip(urls, dests):
        print(f"Scraping {url} to {dest}")
        # Run the scraper
        scrape_season_match_data(url, dest)