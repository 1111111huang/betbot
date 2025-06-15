import pandas as pd

def align_on_keys(feature_df: pd.DataFrame, target_df: pd.DataFrame, key_columns: list[str]):
    """Aligns feature and target DataFrames on key columns, and checks row-wise key alignment."""
    # Check keys exist
    for df_name, df in [('features', feature_df), ('targets', target_df)]:
        missing_keys = [k for k in key_columns if k not in df.columns]
        if missing_keys:
            raise ValueError(f"Missing keys {missing_keys} in {df_name} DataFrame")

    # Merge on key columns
    merged = pd.merge(feature_df, target_df, on=key_columns, how='inner', suffixes=("", "_target"))
    
    if merged.empty:
        raise ValueError("No matching rows found on key columns. Check for mismatches in 'home', 'away', 'date'.")

    # Double-check alignment by comparing key columns row-wise
    for key in key_columns:
        if not (merged[key] == merged[f"{key}"]).all():
            raise ValueError(f"Mismatch detected in key column '{key}' after merging.")

    # Separate aligned outputs
    X_aligned = merged[feature_df.columns]
    target_aligned = merged[target_df.columns]

    return X_aligned.reset_index(drop=True), target_aligned.reset_index(drop=True)
