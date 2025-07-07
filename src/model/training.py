import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
from itertools import product

def train_model_and_collect_metrics(
    feature,
    target,
    target_ranges,
    model_wrapper_class,
    param_grid,  # dict of param_name: list of values, or single dict for one param set
    k,
    key_columns,
    test_size,
    return_final_model=False,
    include_key_columns=False,
    verbose=True
):
    """
    Shared function for cross-validation, metric calculation, and (optionally) final model training.
    feature: numpy array [N, ...]
    target: numpy array [N] or [N, ...]
    """

    n_samples = len(feature)
    test_split = int(n_samples * (1 - test_size))
    X_train, X_test = feature[:test_split], feature[test_split:]
    y_train, y_test = target[:test_split], target[test_split:]

    # Use KFold or TimeSeriesSplit for cross-validation
    cv = TimeSeriesSplit(n_splits=k) if k > 1 else None

    metrics_dict = {}
    best_model = None

    # Prepare param grid as list of dicts
    if isinstance(param_grid, dict) and param_grid:
        from itertools import product
        keys, values = zip(*param_grid.items())
        param_dicts = [dict(zip(keys, v)) for v in product(*values)]
    else:
        param_dicts = [{}]

    for params in param_dicts:
        scores = []
        if cv:
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
                print(f"Training fold {fold_idx + 1}/{k} with params: {params}")
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                model = model_wrapper_class(**params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict_proba(X_val).argmax(axis=1)
                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
                scores.append({'accuracy': acc, 'precision': prec})
        else:
            print(f"Training single split with params: {params}")
            model = model_wrapper_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test).argmax(axis=1)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            scores.append({'accuracy': acc, 'precision': prec})

        # Aggregate metrics
        avg_metrics = {k: np.mean([score[k] for score in scores]) for k in scores[0]}
        metrics_dict[str(params)] = avg_metrics

        if return_final_model:
            best_model = model

    return metrics_dict, best_model

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
    print(f"Using MLflow tracking URI: {tracking_uri}, experiment: {experiment_name}")

    # Use shared function for training and metrics
    param_metrics, final_models = train_model_and_collect_metrics(
        feature_df,
        target_df,
        target_ranges,
        model_wrapper_class,
        model_params,
        k,
        key_columns,
        test_size,
        return_final_model=True,
        random_state=random_state
    )

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

            # Save model
            model = final_models[param_key]
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            model_file_name = f"{run_name}.joblib"
            model_file_path = os.path.join(save_model_path, model_file_name)
            joblib.dump(model, model_file_path)
            mlflow.log_artifact(model_file_path, artifact_path="model")

            print(f"Saved and logged model for {param_data['target_col']} with params: {param_data['params']}")
