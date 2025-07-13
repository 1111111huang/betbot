import os
import mlflow
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
from itertools import product

def compute_threshold_metrics(y_true, y_pred, min_val, max_val):
    """Compute accuracy and precision for lte and gt thresholds."""
    metrics = {}
    for thresh in range(min_val, max_val + 1):
        # lte: true values <= thresh
        mask_lte = y_true <= thresh
        if np.any(mask_lte):
            acc = accuracy_score(y_true[mask_lte], y_pred[mask_lte])
            prec = precision_score(y_true[mask_lte], y_pred[mask_lte], average='macro', zero_division=0)
            metrics[f"lte_{thresh}_accuracy"] = acc
            metrics[f"lte_{thresh}_precision"] = prec
        # gt: true values > thresh
        mask_gt = y_true > thresh
        if np.any(mask_gt):
            acc = accuracy_score(y_true[mask_gt], y_pred[mask_gt])
            prec = precision_score(y_true[mask_gt], y_pred[mask_gt], average='macro', zero_division=0)
            metrics[f"gt_{thresh}_accuracy"] = acc
            metrics[f"gt_{thresh}_precision"] = prec
    return metrics

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

    # Explicit mapping from value to class index
    if isinstance(y_train, pd.Series):
        all_classes = sorted(y_train.unique())
    else:
        all_classes = sorted(np.unique(y_train))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # Encode train/test labels
    if isinstance(y_train, pd.Series):
        y_train_encoded = y_train.map(class_to_idx).values
        y_test_encoded = y_test.map(class_to_idx).values
    else:
        y_train_encoded = np.vectorize(class_to_idx.get)(y_train)
        y_test_encoded = np.vectorize(class_to_idx.get)(y_test)

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
        threshold_metrics = {}
        if cv:
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
                if verbose:
                    print(f"Training fold {fold_idx + 1}/{k} with params: {params}")
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
                model = model_wrapper_class(**params)
                model.fit(X_tr, y_tr)
                y_pred_idx = model.predict_proba(X_val).argmax(axis=1)
                # Map predictions and y_val back to original values
                y_pred = np.vectorize(idx_to_class.get)(y_pred_idx)
                y_val_orig = np.vectorize(idx_to_class.get)(y_val)
                acc = accuracy_score(y_val_orig, y_pred)
                prec = precision_score(y_val_orig, y_pred, average='macro', zero_division=0)
                scores.append({'accuracy': acc, 'precision': prec})

                # Compute threshold metrics for each target
                for target_col, (min_val, max_val) in target_ranges.items():
                    th_metrics = compute_threshold_metrics(y_val_orig, y_pred, min_val, max_val)
                    for k_, v_ in th_metrics.items():
                        threshold_metrics.setdefault(f"{target_col}_{k_}", []).append(v_)
        else:
            if verbose:
                print(f"Training single split with params: {params}")
            model = model_wrapper_class(**params)
            model.fit(X_train, y_train_encoded)
            y_pred_idx = model.predict_proba(X_test).argmax(axis=1)
            y_pred = np.vectorize(idx_to_class.get)(y_pred_idx)
            y_test_orig = np.vectorize(idx_to_class.get)(y_test_encoded)
            acc = accuracy_score(y_test_orig, y_pred)
            prec = precision_score(y_test_orig, y_pred, average='macro', zero_division=0)
            scores.append({'accuracy': acc, 'precision': prec})

            for target_col, (min_val, max_val) in target_ranges.items():
                th_metrics = compute_threshold_metrics(y_test_orig, y_pred, min_val, max_val)
                for k_, v_ in th_metrics.items():
                    threshold_metrics.setdefault(f"{target_col}_{k_}", []).append(v_)

        # Aggregate metrics
        avg_metrics = {k: np.mean([score[k] for score in scores]) for k in scores[0]}
        # Add threshold metrics (mean over folds if CV)
        for k_, v_ in threshold_metrics.items():
            avg_metrics[k_] = np.mean(v_)
            avg_metrics[k_ + "_std"] = np.std(v_)
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
