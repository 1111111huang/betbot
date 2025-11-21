from const import REPO_PATH
from train_config import MODEL_CONFIG
import sys
sys.path.insert(1, f"{REPO_PATH}")

from src.model.training import train_and_log_model
from src.model.experiment_utils import align_on_keys

import pandas as pd

if __name__ == "__main__":
    processed_data_path = MODEL_CONFIG["processed_data_path"]
    target_ranges = MODEL_CONFIG["target_ranges"]

    features_path = f"{MODEL_CONFIG['features_path']}/all_combined_features_2017-24.csv"

    seasons=sorted(MODEL_CONFIG['seasons'])
    target_dfs=[pd.read_csv(f"{MODEL_CONFIG['processed_data_path']}/{season}/all_target_df.csv") for season in seasons[1:]]
    for df in target_dfs:
        df['date']= pd.to_datetime(df['date'])
    target_df = pd.concat(target_dfs, ignore_index=True)

    feature_df = pd.read_csv(features_path)
    feature_df['date']= pd.to_datetime(feature_df['date'])
    feature_df, target_df=align_on_keys(feature_df, target_df, MODEL_CONFIG['key_columns'])

    feature_df.fillna(-1, inplace=True)

    print(feature_df.shape, target_df.shape)

    train_and_log_model(
        feature_df=feature_df,
        target_df=target_df,
        target_ranges=target_ranges,
        model_wrapper_class=MODEL_CONFIG["model_wrapper_class"],
        model_params=MODEL_CONFIG["params"],
        test_size=MODEL_CONFIG.get("test_size", 0.2),
        experiment_name=MODEL_CONFIG["experiment_name"],
        tracking_uri=MODEL_CONFIG["mlflow_tracking_uri"],
        model_name=MODEL_CONFIG["model_name"],
        k=MODEL_CONFIG.get("k", 5),
        key_columns=MODEL_CONFIG["key_columns"],
        save_model_path=MODEL_CONFIG["save_model_path"],
        random_state=MODEL_CONFIG.get("random_state", None)
    )