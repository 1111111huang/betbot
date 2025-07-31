import sys
from build_config import ALL_SEASONS, COMPETITIONS
from const import REPO_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH
from fbref_const import URLs, TARGET_COLUMNS
from build_utils import scrape_competitions_for_seasons, process_raw_match_data



if __name__=='__main__':
    #scrape_competitions_for_seasons(ALL_SEASONS, RAW_DATA_PATH, COMPETITIONS, URLs)
    process_raw_match_data(COMPETITIONS, ALL_SEASONS, RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMNS)