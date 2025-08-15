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
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before calling transform_spot.")
        spot_df = spot_df.copy()
        spot_df[self.date_col] = pd.to_datetime(spot_df[self.date_col])
        spot_df = spot_df.sort_values(self.date_col).reset_index(drop=True)

        # Process all spot data directly using the encoder
        encoded_home = []
        encoded_away = []
        home_categories = set(self.encoder.categories_[0])
        away_categories = set(self.encoder.categories_[1])

        for _, row in spot_df.iterrows():
            home_team = row[self.home_col]
            away_team = row[self.away_col]

            # Check if teams are in the encoder's known categories
            if home_team not in home_categories:
                encoded_home.append('new_team_home')
            else:
                encoded_home.append(home_team)

            if away_team not in away_categories:
                encoded_away.append('new_team_away')
            else:
                encoded_away.append(away_team)

        spot_df['encoded_home'] = encoded_home
        spot_df['encoded_away'] = encoded_away

        # Transform using the encoder
        team_features = self.encoder.transform(spot_df[['encoded_home', 'encoded_away']])
        team_feature_names = self.encoder.get_feature_names_out(['encoded_home', 'encoded_away'])
        team_df = pd.DataFrame(team_features, columns=team_feature_names, index=spot_df.index)

        # Combine with original metadata columns
        meta_df = spot_df[[self.home_col, self.away_col, self.date_col]].reset_index(drop=True)
        team_df = pd.concat([meta_df, team_df.reset_index(drop=True)], axis=1)
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
        spot_df = spot_df.sort_values(self.date_col).reset_index(drop=True)

        # First build complete team history from all available seasons
        team_history = {}
        
        # Get column structure from first available season
        first_season_df = next(iter(season_dfs.values()))
        available_columns = [col for col in first_season_df.columns 
                           if col not in [self.date_col, self.home_col, self.away_col]]

        # Build history from historical data
        for season_key in sorted_keys:
            df = season_dfs[season_key]
            for _, row in df.iterrows():
                match_date = pd.to_datetime(row[self.date_col])
                
                # Only include matches that happened before the earliest spot match
                if match_date >= spot_df[self.date_col].min():
                    continue
                    
                home_team = row[self.home_col]
                away_team = row[self.away_col]
                
                team_history.setdefault(home_team, []).append({
                    'date': match_date,
                    'row': row,
                    'was_home': True
                })
                team_history.setdefault(away_team, []).append({
                    'date': match_date,
                    'row': row,
                    'was_home': False
                })

        # Process spot matches chronologically
        results = []
        for _, row in spot_df.iterrows():
            match_date = pd.to_datetime(row[self.date_col])
            home_team = row[self.home_col]
            away_team = row[self.away_col]

            lag_row = pd.Series(dtype=object)
            lag_row[self.home_col] = home_team
            lag_row[self.away_col] = away_team
            lag_row[self.date_col] = match_date

            # Get history before current match date for each team
            for team_role, team_name in [(self.home_col, home_team), (self.away_col, away_team)]:
                history = [h for h in team_history.get(team_name, []) if h['date'] < match_date]
                history.sort(key=lambda x: x['date'])  # Ensure chronological order
                
                for lag_i in range(1, self.lookback + 1):
                    if len(history) >= lag_i:
                        hist_entry = history[-lag_i]
                        hist_row = hist_entry['row']
                        was_home = hist_entry['was_home']

                        # Copy all columns except identifiers
                        for col in available_columns:
                            lag_row[f'{team_role}_lag{lag_i}_{col}'] = hist_row[col]
                        lag_row[f'{team_role}_lag{lag_i}_was_home'] = int(was_home)
                    else:
                        # Set all features to None when no history exists
                        for col in available_columns:
                            lag_row[f'{team_role}_lag{lag_i}_{col}'] = None
                        lag_row[f'{team_role}_lag{lag_i}_was_home'] = None
            
            # Order columns consistently
            lag_row = lag_row[[self.home_col, self.away_col, self.date_col] + 
                             [c for c in lag_row.index if c not in [self.home_col, self.away_col, self.date_col]]]
            results.append(lag_row)

            # Update history with current match for future spot matches
            match_info = {
                'date': match_date,
                'row': row,
                'was_home': True
            }
            team_history.setdefault(home_team, []).append({**match_info, 'was_home': True})
            team_history.setdefault(away_team, []).append({**match_info, 'was_home': False})

        if results:
            result_df = pd.DataFrame(results)
            result_df[self.date_col] = pd.to_datetime(result_df[self.date_col])
            return result_df
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

        # Find the most recent season in season_dfs that's before each spot season
        spot_df['__season__'] = spot_df[self.date_col].apply(get_season)
        
        # Convert season keys to dates for comparison
        season_dates = {}
        for season in sorted_keys:
            year = int(season.split('-')[0])
            # Use August 1st as the season start date
            season_dates[season] = pd.Timestamp(year=year, month=8, day=1)

        results = []
        for season in spot_df['__season__'].unique():
            # Get the date of the spot season
            spot_year = int(season.split('-')[0])
            spot_season_date = pd.Timestamp(year=spot_year, month=8, day=1)

            # Find the most recent previous season
            prev_season_key = None
            for season_key, season_date in season_dates.items():
                if season_date < spot_season_date and (prev_season_key is None or season_date > season_dates[prev_season_key]):
                    prev_season_key = season_key

            if prev_season_key is None:
                continue  # No previous season found

            # Get spot matches for this season
            sub_spot = spot_df[spot_df['__season__'] == season]
            
            # Compute averages from previous season
            prev_season = season_dfs[prev_season_key]
            team_avg_stats = self._compute_team_averages(prev_season)

            # Create output DataFrame directly from spot matches
            output_df = sub_spot[[self.home_col, self.away_col, self.date_col]].reset_index(drop=True)

            # Add average stats for each team
            for team_role in [self.home_col, self.away_col]:
                role_avg_df = output_df[team_role].map(lambda team: team_avg_stats.get(team, {}))
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
        
        # Initialize last_match_dates with data from all previous seasons
        last_match_dates = defaultdict(lambda: None)
        
        # Process all historical matches to build up last_match_dates
        for season_key in sorted_keys:
            df = season_dfs[season_key]
            for _, row in df.iterrows():
                date = pd.to_datetime(row[self.date_col])
                if date >= spot_df[self.date_col].min():
                    continue
                home_team = row[self.home_col]
                away_team = row[self.away_col]
                last_match_dates[home_team] = date
                last_match_dates[away_team] = date
        
        # Process spot matches
        results = []
        for _, row in spot_df.iterrows():
            date = pd.to_datetime(row[self.date_col])
            home_team = row[self.home_col]
            away_team = row[self.away_col]
            
            last_home_date = last_match_dates[home_team]
            last_away_date = last_match_dates[away_team]
            
            home_rest = (date - last_home_date).days if last_home_date is not None else None
            away_rest = (date - last_away_date).days if last_away_date is not None else None
            
            results.append({
                self.home_col: home_team,
                self.away_col: away_team,
                self.date_col: date,
                'days_since_last_home': home_rest,
                'days_since_last_away': away_rest
            })
            
            # Update last match dates for next matches
            last_match_dates[home_team] = date
            last_match_dates[away_team] = date
        
        if results:
            result_df = pd.DataFrame(results)
            result_df[self.date_col] = pd.to_datetime(result_df[self.date_col])
            return result_df
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
        
        # Convert all dataframes to datetime and sort them
        for k in sorted_keys:
            season_dfs[k][self.date_col] = pd.to_datetime(season_dfs[k][self.date_col])
            season_dfs[k] = season_dfs[k].sort_values(self.date_col).reset_index(drop=True)
            if isinstance(target_dfs, dict):
                target_dfs[k][self.date_col] = pd.to_datetime(target_dfs[k][self.date_col])
                target_dfs[k] = target_dfs[k].sort_values(self.date_col).reset_index(drop=True)
                
        spot_df = spot_df.copy()
        spot_df[self.date_col] = pd.to_datetime(spot_df[self.date_col])
        spot_df = spot_df.sort_values(self.date_col).reset_index(drop=True)
        
        # First build team history from all available seasons' data that occurred before the spot matches
        team_history = {}
        target_cols = []
        
        # Get target columns from first available target_df
        if isinstance(target_dfs, dict):
            first_target_df = next(iter(target_dfs.values()))
        else:
            first_target_df = target_dfs[0]
        target_cols = [col for col in first_target_df.columns if col not in [self.home_col, self.away_col, self.date_col]]
        
        # Build history from historical data
        for season_key in sorted_keys:
            df = season_dfs[season_key]
            if isinstance(target_dfs, dict):
                targets = target_dfs[season_key]
            else:
                idx = sorted_keys.index(season_key)
                targets = target_dfs[idx]
                
            # Merge season data with targets
            # Ensure date columns are datetime before merging
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            targets[self.date_col] = pd.to_datetime(targets[self.date_col])
            merged = df.merge(targets, on=[self.date_col, self.home_col, self.away_col], suffixes=('', '_target'), how='inner')
            
            for _, row in merged.iterrows():
                match_date = pd.to_datetime(row[self.date_col])
                # Only include matches that happened before the earliest spot match
                if match_date >= spot_df[self.date_col].min():
                    continue
                    
                home_team = row[self.home_col]
                away_team = row[self.away_col]
                
                for team_name in [home_team, away_team]:
                    if team_name not in team_history:
                        team_history[team_name] = []
                    team_history[team_name].append({
                        'date': match_date,
                        **{col: row[col] for col in target_cols}
                    })
        
        # Process spot matches
        results = []
        for _, row in spot_df.iterrows():
            match_date = pd.to_datetime(row[self.date_col])
            home_team = row[self.home_col]
            away_team = row[self.away_col]
            
            lag_row = {
                self.home_col: home_team,
                self.away_col: away_team,
                self.date_col: match_date
            }
            
            # Get history before current match date for each team
            for team_role, team_name in [(self.home_col, home_team), (self.away_col, away_team)]:
                history = [h for h in team_history.get(team_name, []) if h['date'] < match_date]
                history.sort(key=lambda x: x['date'])  # Ensure chronological order
                
                for lag_i in range(1, self.lookback + 1):
                    if len(history) >= lag_i:
                        hist_entry = history[-lag_i]
                        for target_col in target_cols:
                            lag_row[f"{team_role}_lag_{lag_i}_{target_col}"] = hist_entry[target_col]
                    else:
                        for target_col in target_cols:
                            lag_row[f"{team_role}_lag_{lag_i}_{target_col}"] = None
                            
            results.append(lag_row)
            
            # Update history with current match (if it exists in target_dfs)
            current_season_idx = next((i for i, season_df in enumerate(season_dfs.values()) 
                                    if match_date in season_df[self.date_col].values), None)
            if current_season_idx is not None:
                if isinstance(target_dfs, dict):
                    current_season_key = list(season_dfs.keys())[current_season_idx]
                    if current_season_key in target_dfs:
                        current_targets = target_dfs[current_season_key]
                        target_row = current_targets[
                            (current_targets[self.date_col] == match_date) &
                            (current_targets[self.home_col] == home_team) &
                            (current_targets[self.away_col] == away_team)
                        ]
                        if not target_row.empty:
                            for team_name in [home_team, away_team]:
                                team_history.setdefault(team_name, []).append({
                                    'date': match_date,
                                    **{col: target_row.iloc[0][col] for col in target_cols}
                                })
        
        if results:
            result_df = pd.DataFrame(results)
            result_df[self.date_col] = pd.to_datetime(result_df[self.date_col])
            return result_df
        else:
            return pd.DataFrame()