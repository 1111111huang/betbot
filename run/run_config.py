REPO_PATH='/Users/tianqihuang/Documents/GitHub/betbot/'
RAW_DATA_PATH=f'{REPO_PATH}/data/raw/premier_league/'
PROCESSED_DATA_PATH=f'{REPO_PATH}/data/processed/premier_league/'
DATA_PROCESSOR_PATH=f'{REPO_PATH}/models/data_processors/'
FEATURES_PATH=f'{REPO_PATH}/data/features/premier_league'

SEASONS=[
    #'2015-16',
    #'2016-17',
    #'2017-18',
    #'2018-19',
    '2019-20',
    '2020-21',
    '2021-22',
    '2022-23',
    '2023-24',
    '2024-25',
]

dests=[
    f"{RAW_DATA_PATH}/{season}/" for season in SEASONS
]

