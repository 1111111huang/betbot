from const import RAW_DATA_PATH

SCRAPER_SEASONS = [
    '2017-18',
    '2018-19',
    '2019-20',
    '2020-21',
    '2021-22',
    '2022-23',
    '2023-24',
    '2024-25',
]

COMPETITION = 'la_liga'

DESTS=[
    f"{RAW_DATA_PATH}/{COMPETITION}/{season}/" for season in SCRAPER_SEASONS
]

SCRAPER_CONFIG = {
    'competition': COMPETITION,
    'seasons': SCRAPER_SEASONS,
    'dests': DESTS
}