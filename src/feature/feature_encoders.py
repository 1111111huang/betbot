import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np
from collections import defaultdict

class TeamEncoder:
    def __init__(self, n_first_matches=5, home_col='home', away_col='away', date_col='date', history=1):
        self.n_first_matches = n_first_matches
        self.home_col = home_col
        self.away_col = away_col
        self.date_col = date_col
        self.history = history  # Number of seasons to use for fitting
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.fitted = False
        self.team_last_seen_season = set()
        self.team_match_counts = {}

    def fit(self, season_dfs):
        all_data = []
        team_match_counts = {}
        team_last_seen_season = set()

        # Use the first `history` seasons for fitting
        fit_seasons = season_dfs[:self.history]

        for season_idx, df in enumerate(fit_seasons):
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

        # Transform the seasons after the first `history` seasons
        transform_seasons = season_dfs[self.history:]

        for season_idx, df in enumerate(transform_seasons):
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

        # Attach home_col, away_col, date_col to the transformed dataframe
        meta_df = final_df[[self.home_col, self.away_col, self.date_col]].reset_index(drop=True)
        team_df = pd.concat([meta_df, team_df.reset_index(drop=True)], axis=1)
        # Ensure date_col is datetime
        team_df[self.date_col] = pd.to_datetime(team_df[self.date_col])
        return team_df

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

class TeamLagFeatureGenerator:
    def __init__(self, lookback=5, date_col='date', home_col='home', away_col='away'):
        self.lookback = lookback
        self.date_col = date_col
        self.home_col = home_col
        self.away_col = away_col

    def transform(self, season_dfs):
        all_rows = []

        for season_df in season_dfs:
            df = season_df.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.sort_values(self.date_col)

            # Track match histories for each team
            team_history = {}

            new_rows = []

            for idx, row in df.iterrows():
                home_team = row[self.home_col]
                away_team = row[self.away_col]
                match_date = row[self.date_col]

                lag_row = pd.Series(dtype=object)
                lag_row[self.home_col] = home_team
                lag_row[self.away_col] = away_team
                lag_row[self.date_col] = match_date

                for team_role, team_name in [(self.home_col, home_team), (self.away_col, away_team)]:
                    history = team_history.get(team_name, [])

                    for lag_i in range(1, self.lookback + 1):
                        if len(history) >= lag_i:
                            hist_entry = history[-lag_i]
                            hist_row = hist_entry['row']
                            was_home = hist_entry['was_home']

                            # Copy all numerical columns except identifiers
                            for col in df.columns:
                                if col not in [self.date_col, self.home_col, self.away_col]:
                                    lag_row[f'{team_role}_lag{lag_i}_{col}'] = hist_row[col]

                            # Add one indicator only per lag
                            lag_row[f'{team_role}_lag{lag_i}_was_home'] = int(was_home)
                        else:
                            for col in df.columns:
                                if col not in [self.date_col, self.home_col, self.away_col]:
                                    lag_row[f'{team_role}_lag{lag_i}_{col}'] = None
                            lag_row[f'{team_role}_lag{lag_i}_was_home'] = None

                # Always keep home_col, away_col, date_col in the output
                lag_row = lag_row[[self.home_col, self.away_col, self.date_col] + [c for c in lag_row.index if c not in [self.home_col, self.away_col, self.date_col]]]

                new_rows.append(lag_row)

                # Store current match in team history
                team_history.setdefault(home_team, []).append({'row': row, 'was_home': True})
                team_history.setdefault(away_team, []).append({'row': row, 'was_home': False})

            all_rows.extend(new_rows)

        result_df = pd.DataFrame(all_rows)
        # Ensure date_col is datetime
        if not result_df.empty:
            result_df[self.date_col] = pd.to_datetime(result_df[self.date_col])
        return result_df
    
class PreviousSeasonTeamAverager:
    def __init__(self, decay_factor=1.0, date_col='date', home_col='home', away_col='away'):
        self.decay_factor = decay_factor
        self.date_col = date_col
        self.home_col = home_col
        self.away_col = away_col

    def _get_numeric_cols(self, df):
        return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != self.date_col]

    def _compute_team_averages(self, df):
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        most_recent_date = df[self.date_col].max()
        df['days_ago'] = (most_recent_date - df[self.date_col]).dt.days
        df['weights'] = np.exp(-self.decay_factor * df['days_ago'])

        numeric_cols = self._get_numeric_cols(df)
        team_stats = {}

        for team_col in [self.home_col, self.away_col]:
            for team in df[team_col].unique():
                team_matches = df[df[team_col] == team]
                if team_matches.empty:
                    continue
                weighted_sum = (team_matches[numeric_cols].multiply(team_matches['weights'], axis=0)).sum()
                total_weight = team_matches['weights'].sum()
                if total_weight == 0:
                    continue
                avg_stats = (weighted_sum / total_weight).to_dict()
                if team not in team_stats:
                    team_stats[team] = avg_stats
                else:
                    # average if seen as both home and away
                    for k, v in avg_stats.items():
                        team_stats[team][k] = (team_stats[team].get(k, 0) + v) / 2

        return team_stats

    def transform(self, season_dfs):
        season_dfs = sorted(season_dfs, key=lambda df: pd.to_datetime(df[self.date_col].iloc[0]))
        results = []

        for i in range(1, len(season_dfs)):
            prev_season = season_dfs[i - 1]
            current_season = season_dfs[i].copy()
            team_avg_stats = self._compute_team_averages(prev_season)

            # Prepare output DataFrame with home_col, away_col, date_col
            output_df = current_season[[self.home_col, self.away_col, self.date_col]].reset_index(drop=True)

            for team_role in [self.home_col, self.away_col]:
                role_avg_df = current_season[team_role].map(lambda team: team_avg_stats.get(team, {}))
                role_avg_df = pd.json_normalize(role_avg_df)
                role_avg_df.columns = [f'prev_season_avg_{team_role}_{col}' for col in role_avg_df.columns]
                output_df = pd.concat([output_df, role_avg_df], axis=1)

            results.append(output_df)
        result_df = pd.concat(results, ignore_index=True)
        # Ensure date_col is datetime
        if not result_df.empty:
            result_df[self.date_col] = pd.to_datetime(result_df[self.date_col])
        return result_df

class TeamRestDaysCalculator:
    def __init__(self, home_col='home', away_col='away', date_col='date'):
        self.home_col = home_col
        self.away_col = away_col
        self.date_col = date_col

    def transform(self, season_dfs):
        all_results = []
        last_match_dates = defaultdict(lambda: None)

        for df in season_dfs:
            df = df.copy()
            df.sort_values(by=self.date_col, inplace=True)

            home_rest_days = []
            away_rest_days = []

            for _, row in df.iterrows():
                date = pd.to_datetime(row[self.date_col])
                home_team = row[self.home_col]
                away_team = row[self.away_col]

                last_home_date = last_match_dates[home_team]
                last_away_date = last_match_dates[away_team]

                home_rest = (date - last_home_date).days if last_home_date is not None else None
                away_rest = (date - last_away_date).days if last_away_date is not None else None

                home_rest_days.append(home_rest)
                away_rest_days.append(away_rest)

                last_match_dates[home_team] = date
                last_match_dates[away_team] = date

            df['days_since_last_home'] = home_rest_days
            df['days_since_last_away'] = away_rest_days
            # Include home_col, away_col, date_col in the output
            all_results.append(df[[self.home_col, self.away_col, self.date_col, 'days_since_last_home', 'days_since_last_away']])
        result_df = pd.concat(all_results, ignore_index=True)
        # Ensure date_col is datetime
        if not result_df.empty:
            result_df[self.date_col] = pd.to_datetime(result_df[self.date_col])
        return result_df