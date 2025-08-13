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
        # Accept dict or list
        if isinstance(season_dfs, dict):
            sorted_keys = sorted(season_dfs.keys())
            season_dfs_list = [season_dfs[k] for k in sorted_keys]
        else:
            season_dfs_list = season_dfs
        # Always convert date_col to datetime and sort each DataFrame by date_col before processing
        season_dfs_list = [df.assign(**{self.date_col: pd.to_datetime(df[self.date_col])}).sort_values(self.date_col).reset_index(drop=True) for df in season_dfs_list]
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before calling transform.")

        all_data = []
        team_match_counts = self.team_match_counts.copy()
        team_last_seen_season = self.team_last_seen_season.copy()

        # Transform the seasons after the first `history` seasons
        transform_seasons = season_dfs_list[self.history:]
        home_categories = set(self.encoder.categories_[0])
        away_categories = set(self.encoder.categories_[1])

        for season_idx, df in enumerate(transform_seasons):
            df = df.copy()
            # Already sorted above
            current_teams = set(df[self.home_col]).union(set(df[self.away_col]))

            encoded_home = []
            encoded_away = []

            for _, row in df.iterrows():
                home_team = row[self.home_col]
                away_team = row[self.away_col]

                team_match_counts.setdefault(home_team, 0)
                team_match_counts.setdefault(away_team, 0)

                if home_team not in home_categories:
                    encoded_home.append('new_team_home')
                else:
                    encoded_home.append(home_team)

                if away_team not in away_categories:
                    encoded_away.append('new_team_away')
                else:
                    encoded_away.append(away_team)
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

    def transform_spot(self, season_dfs, spot_df):
        assert isinstance(season_dfs, dict), "season_dfs must be a dict with season keys for transform_spot"
        # Always convert date_col to datetime and sort each DataFrame by date_col before processing
        sorted_keys = sorted(season_dfs.keys())
        for k in sorted_keys:
            season_dfs[k][self.date_col] = pd.to_datetime(season_dfs[k][self.date_col])
            season_dfs[k] = season_dfs[k].sort_values(self.date_col).reset_index(drop=True)
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before calling transform_spot.")
        spot_df = spot_df.copy()
        spot_df[self.date_col] = pd.to_datetime(spot_df[self.date_col])
        # Determine which season each row in spot_df belongs to
        def get_season(date):
            year = date.year
            if date.month >= 8:
                return f"{year}-{str(year+1)[-2:]}"
            else:
                return f"{year-1}-{str(year)[-2:]}"
        spot_df['__season__'] = spot_df[self.date_col].apply(get_season)
        # Collect features for all relevant seasons
        results = []
        for season in spot_df['__season__'].unique():
            if season not in season_dfs:
                continue
            df = season_dfs[season].copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            sub_spot = spot_df[spot_df['__season__'] == season]
            # Merge to get only the rows in sub_spot
            df = df.merge(sub_spot[[self.date_col, self.home_col, self.away_col]],
                         on=[self.date_col, self.home_col, self.away_col], how='inner')
            # Use the same encoding logic as in transform
            team_match_counts = self.team_match_counts.copy()
            team_last_seen_season = self.team_last_seen_season.copy()
            df = df.sort_values(self.date_col)
            encoded_home = []
            encoded_away = []
            home_categories = set(self.encoder.categories_[0])
            away_categories = set(self.encoder.categories_[1])
            for _, row in df.iterrows():
                home_team = row[self.home_col]
                away_team = row[self.away_col]
                team_match_counts.setdefault(home_team, 0)
                team_match_counts.setdefault(away_team, 0)
                if home_team not in home_categories:
                    encoded_home.append('new_team_home')
                else:
                    encoded_home.append(home_team)
                if away_team not in away_categories:
                    encoded_away.append('new_team_away')
                else:
                    encoded_away.append(away_team)
                team_match_counts[home_team] += 1
                team_match_counts[away_team] += 1
            df['encoded_home'] = encoded_home
            df['encoded_away'] = encoded_away
            team_features = self.encoder.transform(df[['encoded_home', 'encoded_away']])
            team_feature_names = self.encoder.get_feature_names_out(['encoded_home', 'encoded_away'])
            team_df = pd.DataFrame(team_features, columns=team_feature_names, index=df.index)
            meta_df = df[[self.home_col, self.away_col, self.date_col]].reset_index(drop=True)
            team_df = pd.concat([meta_df, team_df.reset_index(drop=True)], axis=1)
            team_df[self.date_col] = pd.to_datetime(team_df[self.date_col])
            results.append(team_df)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

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
        # Accept dict or list
        if isinstance(season_dfs, dict):
            sorted_keys = sorted(season_dfs.keys())
            season_dfs_list = [season_dfs[k] for k in sorted_keys]
        else:
            season_dfs_list = season_dfs
        # Always convert date_col to datetime and sort each DataFrame by date_col before processing
        season_dfs_list = [df.assign(**{self.date_col: pd.to_datetime(df[self.date_col])}).sort_values(self.date_col).reset_index(drop=True) for df in season_dfs_list]
        all_rows = []

        for df in season_dfs_list:
            df = df.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            # Already sorted above
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

    def transform_spot(self, season_dfs, spot_df):
        assert isinstance(season_dfs, dict), "season_dfs must be a dict with season keys for transform_spot"
        # Always convert date_col to datetime and sort each DataFrame by date_col before processing
        sorted_keys = sorted(season_dfs.keys())
        for k in sorted_keys:
            season_dfs[k][self.date_col] = pd.to_datetime(season_dfs[k][self.date_col])
            season_dfs[k] = season_dfs[k].sort_values(self.date_col).reset_index(drop=True)
        spot_df = spot_df.copy()
        spot_df[self.date_col] = pd.to_datetime(spot_df[self.date_col])
        def get_season(date):
            year = date.year
            if date.month >= 8:
                return f"{year}-{str(year+1)[-2:]}"
            else:
                return f"{year-1}-{str(year)[-2:]}"
        spot_df['__season__'] = spot_df[self.date_col].apply(get_season)
        results = []
        for season in spot_df['__season__'].unique():
            if season not in season_dfs:
                continue
            df = season_dfs[season].copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            sub_spot = spot_df[spot_df['__season__'] == season]
            # Only keep matches in sub_spot
            mask = df.set_index([self.date_col, self.home_col, self.away_col]).index.isin(
                sub_spot.set_index([self.date_col, self.home_col, self.away_col]).index)
            team_history = {}
            new_rows = []
            for idx, row in df.iterrows():
                home_team = row[self.home_col]
                away_team = row[self.away_col]
                match_date = row[self.date_col]
                if not ((match_date, home_team, away_team) in sub_spot.set_index([self.date_col, self.home_col, self.away_col]).index):
                    team_history.setdefault(home_team, []).append({'row': row, 'was_home': True})
                    team_history.setdefault(away_team, []).append({'row': row, 'was_home': False})
                    continue
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
                            for col in df.columns:
                                if col not in [self.date_col, self.home_col, self.away_col]:
                                    lag_row[f'{team_role}_lag{lag_i}_{col}'] = hist_row[col]
                            lag_row[f'{team_role}_lag{lag_i}_was_home'] = int(was_home)
                        else:
                            for col in df.columns:
                                if col not in [self.date_col, self.home_col, self.away_col]:
                                    lag_row[f'{team_role}_lag{lag_i}_{col}'] = None
                            lag_row[f'{team_role}_lag{lag_i}_was_home'] = None
                lag_row = lag_row[[self.home_col, self.away_col, self.date_col] + [c for c in lag_row.index if c not in [self.home_col, self.away_col, self.date_col]]]
                new_rows.append(lag_row)
                team_history.setdefault(home_team, []).append({'row': row, 'was_home': True})
                team_history.setdefault(away_team, []).append({'row': row, 'was_home': False})
            result_df = pd.DataFrame(new_rows)
            if not result_df.empty:
                result_df[self.date_col] = pd.to_datetime(result_df[self.date_col])
            results.append(result_df)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
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
        # Accept dict or list
        if isinstance(season_dfs, dict):
            sorted_keys = sorted(season_dfs.keys())
            season_dfs_list = [season_dfs[k] for k in sorted_keys]
        else:
            season_dfs_list = season_dfs
        # Always convert date_col to datetime and sort each DataFrame by date_col before processing
        season_dfs_list = [df.assign(**{self.date_col: pd.to_datetime(df[self.date_col])}).sort_values(self.date_col).reset_index(drop=True) for df in season_dfs_list]
        season_dfs_list = sorted(season_dfs_list, key=lambda df: pd.to_datetime(df[self.date_col].iloc[0]))
        results = []

        for i in range(1, len(season_dfs_list)):
            prev_season = season_dfs_list[i - 1]
            current_season = season_dfs_list[i].copy()
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

    def transform_spot(self, season_dfs, spot_df):
        assert isinstance(season_dfs, dict), "season_dfs must be a dict with season keys for transform_spot"
        # Always convert date_col to datetime and sort each DataFrame by date_col before processing
        sorted_keys = sorted(season_dfs.keys())
        for k in sorted_keys:
            season_dfs[k][self.date_col] = pd.to_datetime(season_dfs[k][self.date_col])
            season_dfs[k] = season_dfs[k].sort_values(self.date_col).reset_index(drop=True)
        spot_df = spot_df.copy()
        spot_df[self.date_col] = pd.to_datetime(spot_df[self.date_col])
        def get_season(date):
            year = date.year
            if date.month >= 8:
                return f"{year}-{str(year+1)[-2:]}"
            else:
                return f"{year-1}-{str(year)[-2:]}"
        spot_df['__season__'] = spot_df[self.date_col].apply(get_season)
        results = []
        for season in spot_df['__season__'].unique():
            if season not in season_dfs:
                continue
            season_idx = sorted_keys.index(season)
            if season_idx == 0:
                continue  # No previous season to average from
            prev_season_key = sorted_keys[season_idx - 1]
            prev_season = season_dfs[prev_season_key]
            current_season = season_dfs[season].copy()
            current_season[self.date_col] = pd.to_datetime(current_season[self.date_col])
            sub_spot = spot_df[spot_df['__season__'] == season]
            # Only keep matches in sub_spot
            current_season = current_season.merge(sub_spot[[self.date_col, self.home_col, self.away_col]],
                                                  on=[self.date_col, self.home_col, self.away_col], how='inner')
            team_avg_stats = self._compute_team_averages(prev_season)
            output_df = current_season[[self.home_col, self.away_col, self.date_col]].reset_index(drop=True)
            for team_role in [self.home_col, self.away_col]:
                role_avg_df = current_season[team_role].map(lambda team: team_avg_stats.get(team, {}))
                role_avg_df = pd.json_normalize(role_avg_df)
                role_avg_df.columns = [f'prev_season_avg_{team_role}_{col}' for col in role_avg_df.columns]
                output_df = pd.concat([output_df, role_avg_df], axis=1)
            if not output_df.empty:
                output_df[self.date_col] = pd.to_datetime(output_df[self.date_col])
            results.append(output_df)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

class TeamRestDaysCalculator:
    def __init__(self, home_col='home', away_col='away', date_col='date'):
        self.home_col = home_col
        self.away_col = away_col
        self.date_col = date_col

    def transform(self, season_dfs):
        # Accept dict or list
        if isinstance(season_dfs, dict):
            sorted_keys = sorted(season_dfs.keys())
            season_dfs_list = [season_dfs[k] for k in sorted_keys]
        else:
            season_dfs_list = season_dfs
        # Always convert date_col to datetime and sort each DataFrame by date_col before processing
        season_dfs_list = [df.assign(**{self.date_col: pd.to_datetime(df[self.date_col])}).sort_values(self.date_col).reset_index(drop=True) for df in season_dfs_list]
        all_results = []
        last_match_dates = defaultdict(lambda: None)

        for df in season_dfs_list:
            df = df.copy()
            # Already sorted above
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

    def transform_spot(self, season_dfs, spot_df):
        assert isinstance(season_dfs, dict), "season_dfs must be a dict with season keys for transform_spot"
        # Always convert date_col to datetime and sort each DataFrame by date_col before processing
        sorted_keys = sorted(season_dfs.keys())
        for k in sorted_keys:
            season_dfs[k][self.date_col] = pd.to_datetime(season_dfs[k][self.date_col])
            season_dfs[k] = season_dfs[k].sort_values(self.date_col).reset_index(drop=True)
        spot_df = spot_df.copy()
        spot_df[self.date_col] = pd.to_datetime(spot_df[self.date_col])
        def get_season(date):
            year = date.year
            if date.month >= 8:
                return f"{year}-{str(year+1)[-2:]}"
            else:
                return f"{year-1}-{str(year)[-2:]}"
        spot_df['__season__'] = spot_df[self.date_col].apply(get_season)
        results = []
        for season in spot_df['__season__'].unique():
            if season not in season_dfs:
                continue
            df = season_dfs[season].copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            sub_spot = spot_df[spot_df['__season__'] == season]
            all_results = []
            last_match_dates = defaultdict(lambda: None)
            for _, row in df.iterrows():
                date = pd.to_datetime(row[self.date_col])
                home_team = row[self.home_col]
                away_team = row[self.away_col]
                last_home_date = last_match_dates[home_team]
                last_away_date = last_match_dates[away_team]
                home_rest = (date - last_home_date).days if last_home_date is not None else None
                away_rest = (date - last_away_date).days if last_away_date is not None else None
                if ((date, home_team, away_team) in sub_spot.set_index([self.date_col, self.home_col, self.away_col]).index):
                    all_results.append({
                        self.home_col: home_team,
                        self.away_col: away_team,
                        self.date_col: date,
                        'days_since_last_home': home_rest,
                        'days_since_last_away': away_rest
                    })
                last_match_dates[home_team] = date
                last_match_dates[away_team] = date
            result_df = pd.DataFrame(all_results)
            if not result_df.empty:
                result_df[self.date_col] = pd.to_datetime(result_df[self.date_col])
            results.append(result_df)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

class TeamLagTargetFeature:
    def __init__(self, lookback=5, date_col='date', home_col='home', away_col='away'):
        """
        Args:
            lookback (int): Number of previous matches to use for lag features.
            date_col, home_col, away_col: Column names.
        """
        self.lookback = lookback
        self.date_col = date_col
        self.home_col = home_col
        self.away_col = away_col

    def transform(self, season_dfs, target_dfs):
        # Accept dict or list
        if isinstance(season_dfs, dict):
            sorted_keys = sorted(season_dfs.keys())
            season_dfs_list = [season_dfs[k] for k in sorted_keys]
            if isinstance(target_dfs, dict):
                target_dfs_list = [target_dfs[k] for k in sorted_keys]
            else:
                target_dfs_list = target_dfs
        else:
            season_dfs_list = season_dfs
            target_dfs_list = target_dfs
        # Always convert date_col to datetime and sort each DataFrame by date_col before processing
        season_dfs_list = [df.assign(**{self.date_col: pd.to_datetime(df[self.date_col])}).sort_values(self.date_col).reset_index(drop=True) for df in season_dfs_list]
        target_dfs_list = [df.assign(**{self.date_col: pd.to_datetime(df[self.date_col])}).sort_values(self.date_col).reset_index(drop=True) for df in target_dfs_list]
        all_rows = []

        for df, targets in zip(season_dfs_list, target_dfs_list):
            df = df.copy()
            targets = targets.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            targets[self.date_col] = pd.to_datetime(targets[self.date_col])
            # Align on date, home, away
            merged = df.merge(targets, on=[self.date_col, self.home_col, self.away_col], suffixes=('', '_target'), how='inner')
            # Identify target columns
            target_cols = [col for col in targets.columns if col not in [self.home_col, self.away_col, self.date_col]]
            # Track target history for each team
            team_history = {}
            for idx, row in merged.iterrows():
                home_team = row[self.home_col]
                away_team = row[self.away_col]
                match_date = row[self.date_col]
                # Update team history with current match's target values BEFORE calculating lag features
                for team_role, team_name in [(self.home_col, home_team), (self.away_col, away_team)]:
                    if team_name not in team_history:
                        team_history[team_name] = []
                    team_history[team_name].append({col: row[col] for col in target_cols})
                lag_row = {
                    self.home_col: home_team,
                    self.away_col: away_team,
                    self.date_col: match_date
                }
                for team_role, team_name in [(self.home_col, home_team), (self.away_col, away_team)]:
                    history = team_history.get(team_name, [])
                    # Exclude the current match from its own history
                    history = history[:-1] if len(history) > 0 else []
                    for lag_i in range(1, self.lookback + 1):
                        if len(history) >= lag_i:
                            hist_entry = history[-lag_i]
                            for target_col in target_cols:
                                lag_row[f"{team_role}_lag_{lag_i}_{target_col}"] = hist_entry[target_col]
                        else:
                            for target_col in target_cols:
                                lag_row[f"{team_role}_lag_{lag_i}_{target_col}"] = None
                all_rows.append(lag_row)
        result_df = pd.DataFrame(all_rows)
        if not result_df.empty:
            result_df[self.date_col] = pd.to_datetime(result_df[self.date_col])
        return result_df

    def transform_spot(self, season_dfs, target_dfs, spot_df):
        assert isinstance(season_dfs, dict), "season_dfs must be a dict with season keys for transform_spot"
        if isinstance(target_dfs, dict):
            assert set(target_dfs.keys()).issuperset(season_dfs.keys()), "target_dfs must have at least the same keys as season_dfs"
        # Always convert date_col to datetime and sort each DataFrame by date_col before processing
        sorted_keys = sorted(season_dfs.keys())
        for k in sorted_keys:
            season_dfs[k][self.date_col] = pd.to_datetime(season_dfs[k][self.date_col])
            season_dfs[k] = season_dfs[k].sort_values(self.date_col).reset_index(drop=True)
            if isinstance(target_dfs, dict):
                target_dfs[k][self.date_col] = pd.to_datetime(target_dfs[k][self.date_col])
                target_dfs[k] = target_dfs[k].sort_values(self.date_col).reset_index(drop=True)
        spot_df = spot_df.copy()
        spot_df[self.date_col] = pd.to_datetime(spot_df[self.date_col])
        def get_season(date):
            year = date.year
            if date.month >= 8:
                return f"{year}-{str(year+1)[-2:]}"
            else:
                return f"{year-1}-{str(year)[-2:]}"
        spot_df['__season__'] = spot_df[self.date_col].apply(get_season)
        results = []
        for season in spot_df['__season__'].unique():
            if season not in season_dfs:
                continue
            df = season_dfs[season].copy()
            if isinstance(target_dfs, dict):
                targets = target_dfs[season].copy()
            else:
                idx = sorted_keys.index(season)
                targets = target_dfs[idx].copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            targets[self.date_col] = pd.to_datetime(targets[self.date_col])
            sub_spot = spot_df[spot_df['__season__'] == season]
            # Align on date, home, away for target columns only
            merged = df.merge(targets, on=[self.date_col, self.home_col, self.away_col], suffixes=('', '_target'), how='inner')
            # Only keep matches in sub_spot for output, but build team_history from all matches in merged
            target_cols = [col for col in targets.columns if col not in [self.home_col, self.away_col, self.date_col]]
            team_history = {}
            all_rows = []
            # Build a set of spot matches for quick lookup
            spot_idx = set(sub_spot.set_index([self.date_col, self.home_col, self.away_col]).index)
            for idx, row in merged.iterrows():
                home_team = row[self.home_col]
                away_team = row[self.away_col]
                match_date = row[self.date_col]
                # Always update team history for all matches BEFORE calculating lag features
                for team_role, team_name in [(self.home_col, home_team), (self.away_col, away_team)]:
                    if team_name not in team_history:
                        team_history[team_name] = []
                    team_history[team_name].append({col: row[col] for col in target_cols+[self.date_col, self.home_col, self.away_col]})
                is_spot = (match_date, home_team, away_team) in spot_idx
                if is_spot:
                    lag_row = {
                        self.home_col: home_team,
                        self.away_col: away_team,
                        self.date_col: match_date
                    }
                    for team_role, team_name in [(self.home_col, home_team), (self.away_col, away_team)]:
                        history = team_history.get(team_name, [])
                        # Exclude the current match from its own history
                        history = history[:-1] if len(history) > 0 else []
                        for lag_i in range(1, self.lookback + 1):
                            if len(history) >= lag_i:
                                hist_entry = history[-lag_i]
                                for target_col in target_cols:
                                    lag_row[f"{team_role}_lag_{lag_i}_{target_col}"] = hist_entry[target_col]
                            else:
                                for target_col in target_cols:
                                    lag_row[f"{team_role}_lag_{lag_i}_{target_col}"] = None
                    all_rows.append(lag_row)
            result_df = pd.DataFrame(all_rows)
            if not result_df.empty:
                result_df[self.date_col] = pd.to_datetime(result_df[self.date_col])
            results.append(result_df)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()