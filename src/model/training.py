import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score

def train_and_log_model(
    feature_df,
    target_df,
    target_ranges,
    model_wrapper_class,
    model_params,
    test_size,
    experiment_name,
    tracking_uri,
    model_name,
    k,
    key_columns,
    save_model_path,
    random_state=None
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Sort by date for time series split
    if not feature_df["date"].is_monotonic_increasing:
        raise ValueError("feature_df['date'] must be sorted in increasing order.")
    if not target_df["date"].is_monotonic_increasing:
        raise ValueError("target_df['date'] must be sorted in increasing order.")

    feature_df = feature_df.drop(columns=key_columns, errors='ignore')
    target_df = target_df.drop(columns=key_columns, errors='ignore')

    n_samples = len(feature_df)
    test_split = int(n_samples * (1 - test_size))
    train_valid_df = feature_df.iloc[:test_split]
    test_df = feature_df.iloc[test_split:]
    train_valid_target = target_df.iloc[:test_split]
    test_target = target_df.iloc[test_split:]

    n_train_valid = len(train_valid_df)
    min_train_size = n_train_valid // 2
    fold_size = (n_train_valid - min_train_size) // k

    tscv = TimeSeriesSplit(n_splits=k, test_size=fold_size)

    param_metrics = {}

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(train_valid_df)):
        print(f"\nTraining fold {fold+1}/{k}...")
        X_train = train_valid_df.iloc[train_idx]
        y_train_df = train_valid_target.iloc[train_idx]
        X_valid = train_valid_df.iloc[valid_idx]
        y_valid_df = train_valid_target.iloc[valid_idx]
        X_test = test_df
        y_test_df = test_target

        for target_col, (lower_bound, upper_bound) in target_ranges.items():
            print(f"Training target variable: {target_col}")
            y_train = y_train_df[target_col]
            y_valid = y_valid_df[target_col]
            y_test = y_test_df[target_col]

            all_classes = sorted(y_train.unique())
            class_to_idx = {c: i for i, c in enumerate(all_classes)}
            idx_to_class = {i: c for c, i in class_to_idx.items()}

            y_train_encoded = y_train.map(class_to_idx)
            y_valid_encoded = y_valid.map(class_to_idx)
            y_test_encoded = y_test.map(class_to_idx)

            # Only one param set
            param_key = (target_col, tuple(model_params.items()))

            if param_key not in param_metrics:
                param_metrics[param_key] = {
                    'target_col': target_col,
                    'params': model_params,
                    'metrics': defaultdict(list)
                }

            model = model_wrapper_class(**model_params)
            model.fit(X_train, y_train_encoded)

            for split_name, X_split, y_split, y_split_encoded in [
                ("train", X_train, y_train, y_train_encoded),
                ("valid", X_valid, y_valid, y_valid_encoded),
                ("test", X_test, y_test, y_test_encoded)
            ]:
                proba = model.predict_proba(X_split)
                predicted_class_indices = np.argmax(proba, axis=1)
                predicted_classes = np.array([idx_to_class[i] for i in predicted_class_indices])

                base_acc = accuracy_score(y_split, predicted_classes)
                param_metrics[param_key]['metrics'][f"{target_col}_{split_name}_top1_accuracy"].append(base_acc)

                for threshold in range(lower_bound, upper_bound + 1):
                    y_true_binary = (y_split > threshold).astype(int)
                    proba_gt = np.zeros(len(y_split))
                    for cls, idx in class_to_idx.items():
                        if cls > threshold:
                            proba_gt += proba[:, idx]
                    y_pred_binary = (proba_gt >= 0.5).astype(int)
                    acc = accuracy_score(y_true_binary, y_pred_binary)
                    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                    param_metrics[param_key]['metrics'][f"{target_col}_{split_name}_gt_{threshold}_accuracy"].append(acc)
                    param_metrics[param_key]['metrics'][f"{target_col}_{split_name}_gt_{threshold}_precision"].append(prec)

                y_true_le = (y_split <= lower_bound).astype(int)
                proba_gt_lb = np.zeros(len(y_split))
                for cls, idx in class_to_idx.items():
                    if cls > lower_bound:
                        proba_gt_lb += proba[:, idx]
                proba_le = 1 - proba_gt_lb
                y_pred_le = (proba_le >= 0.5).astype(int)
                acc_le = accuracy_score(y_true_le, y_pred_le)
                prec_le = precision_score(y_true_le, y_pred_le, zero_division=0)
                param_metrics[param_key]['metrics'][f"{target_col}_{split_name}_lte_{lower_bound}_accuracy"].append(acc_le)
                param_metrics[param_key]['metrics'][f"{target_col}_{split_name}_lte_{lower_bound}_precision"].append(prec_le)

    # Log average metrics across folds for each parameter combination and save model
    for param_key, param_data in param_metrics.items():
        run_name = f"{model_name}_{param_data['target_col']}_" + "_".join(f"{k}={v}" for k, v in param_data['params'].items())
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("target", param_data['target_col'])
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(param_data['params'])

            for metric_name, values in param_data['metrics'].items():
                avg_value = np.mean(values)
                std_value = np.std(values)
                mlflow.log_metric(f"{metric_name}", avg_value)
                mlflow.log_metric(f"{metric_name}_std", std_value)

            # Fit on all train_valid data for final model saving
            model = model_wrapper_class(**param_data['params'])
            y_full = train_valid_target[param_data['target_col']]
            all_classes = sorted(y_full.unique())
            class_to_idx = {c: i for i, c in enumerate(all_classes)}
            y_full_encoded = y_full.map(class_to_idx)
            model.fit(train_valid_df, y_full_encoded)

            # Save model
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            joblib.dump(model, save_model_path)
            mlflow.log_artifact(save_model_path, artifact_path="model")

            print(f"Saved and logged model for {param_data['target_col']} with params: {param_data['params']}")
