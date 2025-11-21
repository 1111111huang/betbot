import pandas as pd
from datetime import datetime

def read_epl_data_by_season(seasons, data_path):
    all_data_df={}
    all_target_df={}
    for season in seasons:
        season_path=f'{data_path}/{season}/'
        all_data_df[season]=pd.read_csv(f'{season_path}/all_data_df.csv')
        all_data_df[season]['date']=all_data_df[season]['date'].apply(lambda s: datetime.strptime(s, '%B-%d-%Y'))
        all_data_df[season]=all_data_df[season].sort_values('date').reset_index(drop=True)

        all_target_df[season]=pd.read_csv(f'{season_path}/all_target_df.csv')
        all_target_df[season]['date']=all_target_df[season]['date'].apply(lambda s: datetime.strptime(s, '%B-%d-%Y'))
        all_target_df[season]=all_target_df[season].sort_values('date').reset_index(drop=True)
    
    all_data_df=pd.concat((all_data_df.values()), axis=0)
    all_data_df=all_data_df.set_index(['date','home','away'])
 
    all_target_df=pd.concat(all_target_df.values(), axis=0)
    all_target_df=all_target_df.set_index(['date','home','away'])
    
    return all_data_df, all_target_df

def compute_match_day_diff(all_data_df):
    seasons=all_data_df.keys()
    for season in seasons:
        all_teams=all_data_df[season]['home'].unique().tolist()
        all_team_date_dff=[]
        team_temp_counter={}
        for team in all_teams:
            team_temp_counter[team]=0
            team_data_df=all_data_df[season][(all_data_df[season]['home']==team)|(all_data_df[season]['away']==team)]
            date_diff=team_data_df['date'].iloc[1:].reset_index(drop=True)-team_data_df['date'].iloc[:-1].reset_index(drop=True)
            date_diff=date_diff.apply(lambda x: x.days)
            date_diff.index=date_diff.index+1
            date_diff[0]=-1
            date_diff=date_diff.sort_index()
            all_team_date_dff.append(date_diff)
        all_team_date_dff=pd.concat(all_team_date_dff, axis=1)
        all_team_date_dff.columns=all_teams

        home_day_diff, away_day_diff=[], []
        for _, row in all_data_df[season].iterrows():
            home, away=row['home'], row['away']
            home_day_diff.append(all_team_date_dff.loc[team_temp_counter[home], home])
            team_temp_counter[home]+=1
            away_day_diff.append(all_team_date_dff.loc[team_temp_counter[away], away])
            team_temp_counter[away]+=1
        all_data_df[season]['home_day_diff']=home_day_diff
        all_data_df[season]['away_day_diff']=away_day_diff
    return all_data_df

def compute_historical_feature(feature_df, history=5):
    historical_feature=[]
    hist_ind=[]
    for ind, row in feature_df.iterrows():
        match_feature=[]
        home_team, away_team = row['home'], row['away']
        home_match_ix=team_matches[home_team].index(ind)
        away_match_ix=team_matches[away_team].index(ind)
        if home_match_ix<history or away_match_ix<history:
            continue
        for i in range(history):
            match_feature.append(feature_df.loc[team_matches[home_team][home_match_ix-1-i]].drop(['home', 'away', 'date']).rename(lambda x: f'home_{x}_{-i-1}'))
            match_feature.append(feature_df.loc[team_matches[away_team][away_match_ix-1-i]].drop(['home', 'away', 'date']).rename(lambda x: f'away_{x}_{-i-1}'))
        match_feature=pd.concat(match_feature)
        historical_feature.append(match_feature)
        hist_ind.append(ind)
    historical_feature=pd.DataFrame(historical_feature)
    historical_feature.index=hist_ind
    return historical_feature

