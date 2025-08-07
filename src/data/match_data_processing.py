import pandas as pd
import re
pattern=r"(?:January|February|March|April|May|June|July|August|September|October|November|December)-\d{1,2}-\d{4}"

def process_shot_stat_df(df):
    shot_stat_df=pd.DataFrame(df)
    shot_stat_df.columns=[f'{col_pair[1]}'.lower() for col_pair in shot_stat_df.columns.tolist() if 'Unnamed: ' in col_pair[0]]+[f'{col_pair[0]}_{col_pair[1]}'.replace(' ', '_').lower() for col_pair in shot_stat_df.columns.tolist() if 'Unnamed: ' not in col_pair[0]]
    return shot_stat_df

def process_player_stat_df(df):
    player_df=pd.DataFrame(df)
    cols=[f'player_{col_pair[1]}'.lower() for col_pair in player_df.columns.tolist() if 'Unnamed: ' in col_pair[0]]+[f'{col_pair[0]}_{col_pair[1]}'.replace(' ', '_').lower() for col_pair in player_df.columns.tolist() if 'Unnamed: ' not in col_pair[0]]
    player_df.columns=cols
    player_df=player_df.drop(columns=['player_#', 'player_nation', 'player_pos', 'player_age', 'player_min']).rename(columns={'player_player': 'player'})
    return player_df

def process_match_stat_df(df):
    match_stat_df=df
    match_stat_df=match_stat_df.transpose().reset_index(level=1).transpose().reset_index(drop=True)
    match_stat_dict=dict([(match_stat_df.iloc[i*2, 0], [match_stat_df.iloc[i*2+1,0], match_stat_df.iloc[i*2+1,1]]) for i in range(int(len(match_stat_df)/2))])
    match_stat_dict['Possession']=[match_stat_dict['Possession'][0][:-1], match_stat_dict['Possession'][1][:-1]]
    match_stat_dict['passes']=[match_stat_dict['Passing Accuracy'][0].split('\xa0—\xa0')[0].split(' of ')[1], match_stat_dict['Passing Accuracy'][1].split('\xa0—\xa0')[1].split(' of ')[1]]
    match_stat_dict['pass_accuracy']=[match_stat_dict['Passing Accuracy'][0].split('\xa0—\xa0')[1], match_stat_dict['Passing Accuracy'][1].split('\xa0—\xa0')[0]]
    match_stat_dict['pass_accuracy']=[match_stat_dict['pass_accuracy'][0][:-1], match_stat_dict['pass_accuracy'][1][:-1]]
    match_stat_dict['shots']=[match_stat_dict['Shots on Target'][0].split('\xa0—\xa0')[0].split(' of ')[1], match_stat_dict['Shots on Target'][1].split('\xa0—\xa0')[1].split(' of ')[1]]
    match_stat_dict['shots_on_target']=[match_stat_dict['Shots on Target'][0].split('\xa0—\xa0')[0].split(' of ')[0], match_stat_dict['Shots on Target'][1].split('\xa0—\xa0')[1].split(' of ')[0]]
    match_stat_df=pd.DataFrame(match_stat_dict, index=match_stat_df.columns).drop(columns=['Passing Accuracy', 'Shots on Target', 'Saves', 'Cards'])
    match_stat_df.columns=[col.lower().replace(' ','_') for col in match_stat_df.columns.tolist()]
    return match_stat_df

def process_match_data(dfs):
    match_stat_df=process_match_stat_df(dfs[2])
    home_player_stat_df=[process_player_stat_df(df).set_index('player').sort_index() for df in dfs[3:9]]
    home_player_stat_df=pd.concat(home_player_stat_df, axis=1)
    away_player_stat_df=[process_player_stat_df(df).set_index('player').sort_index() for df in dfs[10:16]]
    away_player_stat_df=pd.concat(away_player_stat_df, axis=1)
    shot_stat_df=process_shot_stat_df(dfs[17])

    return match_stat_df, home_player_stat_df, away_player_stat_df, shot_stat_df

def process_match_target_var(home_df, away_df, match_df, match_name):
    assert 'Players' in home_df.iloc[0, 0]
    assert 'Players' in away_df.iloc[0, 0]
    home_ind=home_df.iloc[0, 0]
    away_ind=away_df.iloc[0, 0]
    home_df=home_df.set_index('player')
    away_df=away_df.set_index('player')
    targets={
        'home_goals': home_df.loc[home_ind, 'performance_gls']+away_df.loc[away_ind, 'performance_og'],
        'away_goals': away_df.loc[away_ind, 'performance_gls']+home_df.loc[home_ind, 'performance_og'],
        'home_corners': home_df.loc[home_ind, 'pass_types_ck'],
        'away_corners': away_df.loc[away_ind, 'pass_types_ck'],
        'home_cards': home_df.loc[home_ind, 'performance_crdy'],
        'away_cards': away_df.loc[away_ind, 'performance_crdy'],
        'home_shots': home_df.loc[home_ind, 'performance_sh'],
        'away_shots': away_df.loc[away_ind, 'performance_sh'],
        'home_sots': home_df.loc[home_ind, 'performance_sot'],
        'away_sots': away_df.loc[away_ind, 'performance_sot'],
    }
    targets_df=pd.DataFrame(pd.Series(targets)).transpose()
    targets_df['home']=match_df.index.tolist()[0]
    targets_df['away']=match_df.index.tolist()[1]
    match_date=re.findall(pattern, match_name, re.IGNORECASE)[0]
    targets_df['date']=[match_date]
    return targets_df

def process_match_other_var(home_df, away_df, match_df, match_name, target_columns):
    assert 'Players' in home_df.iloc[0, 0]
    assert 'Players' in away_df.iloc[0, 0]
    home_df=home_df.drop(columns=target_columns).rename(columns=lambda x: f'home_{x}').head(1)
    away_df=away_df.drop(columns=target_columns).rename(columns=lambda x: f'away_{x}').head(1)
    data_df=pd.concat([home_df, away_df], axis=1)
    data_df['home']=[match_df.index.tolist()[0]]
    data_df['away']=[match_df.index.tolist()[1]]
    match_date=re.findall(pattern, match_name, re.IGNORECASE)[0]
    data_df['date']=[match_date]
    return data_df
    