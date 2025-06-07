import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

class TeamEncoder:
    def __init__(self, n_first_matches=5, home_col='home', away_col='away', date_col='date'):
        self.n_first_matches = n_first_matches
        self.home_col = home_col
        self.away_col = away_col
        self.date_col = date_col
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.fitted = False
        self.team_last_seen_season = set()
        self.team_match_counts = {}

    def fit(self, season_dfs):
        all_data = []
        team_match_counts = {}
        team_last_seen_season = set()

        for season_idx, df in enumerate(season_dfs):
            df = df.copy()
            df.sort_values(by=self.date_col, inplace=True)
            current_teams = set(df[self.home_col]).union(set(df[self.away_col]))

            encoded_home = []
            encoded_away = []

            for _, row in df.iterrows():
                home_team = row[self.home_col]
                away_team = row[self.away_col]

                home_new = home_team not in team_last_seen_season
                away_new = away_team not in team_last_seen_season

                team_match_counts.setdefault(home_team, 0)
                team_match_counts.setdefault(away_team, 0)

                if home_new and team_match_counts[home_team] < self.n_first_matches:
                    encoded_home.append('new_team_home')
                else:
                    encoded_home.append(home_team)

                if away_new and team_match_counts[away_team] < self.n_first_matches:
                    encoded_away.append('new_team_away')
                else:
                    encoded_away.append(away_team)

                team_match_counts[home_team] += 1
                team_match_counts[away_team] += 1

            df['encoded_home'] = encoded_home
            df['encoded_away'] = encoded_away
            all_data.append(df)

            team_last_seen_season = current_teams

        final_df = pd.concat(all_data, ignore_index=True)
        self.encoder.fit(final_df[['encoded_home', 'encoded_away']])
        self.team_last_seen_season = team_last_seen_season
        self.team_match_counts = team_match_counts
        self.fitted = True
        return self

    def transform(self, season_dfs):
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before calling transform.")
        all_data = []
        team_match_counts = self.team_match_counts.copy()
        team_last_seen_season = self.team_last_seen_season.copy()

        for season_idx, df in enumerate(season_dfs):
            df = df.copy()
            df.sort_values(by=self.date_col, inplace=True)
            current_teams = set(df[self.home_col]).union(set(df[self.away_col]))

            encoded_home = []
            encoded_away = []

            for _, row in df.iterrows():
                home_team = row[self.home_col]
                away_team = row[self.away_col]

                home_new = home_team not in team_last_seen_season
                away_new = away_team not in team_last_seen_season

                team_match_counts.setdefault(home_team, 0)
                team_match_counts.setdefault(away_team, 0)

                if home_new and team_match_counts[home_team] < self.n_first_matches:
                    encoded_home.append('new_team_home')
                else:
                    encoded_home.append(home_team)

                if away_new and team_match_counts[away_team] < self.n_first_matches:
                    encoded_away.append('new_team_away')
                else:
                    encoded_away.append(away_team)

                team_match_counts[home_team] += 1
                team_match_counts[away_team] += 1

            df['encoded_home'] = encoded_home
            df['encoded_away'] = encoded_away
            all_data.append(df)
            team_last_seen_season = current_teams

        final_df = pd.concat(all_data, ignore_index=True)
        team_features = self.encoder.transform(final_df[['encoded_home', 'encoded_away']])
        team_feature_names = self.encoder.get_feature_names_out(['encoded_home', 'encoded_away'])
        team_df = pd.DataFrame(team_features, columns=team_feature_names, index=final_df.index)

        return pd.concat([final_df, team_df], axis=1)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)