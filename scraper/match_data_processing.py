import pandas as pd

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
    
    