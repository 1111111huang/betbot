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
        if thresh == min_val:
            # lte: true values <= thresh
            mask_lte = y_true <= thresh
            if np.any(mask_lte):
                acc = accuracy_score(y_true[mask_lte], y_pred[mask_lte])
                prec = precision_score(y_true[mask_lte], y_pred[mask_lte], average='macro', zero_division=0.0)
                metrics[f"lte_{thresh}_accuracy"] = acc
                metrics[f"lte_{thresh}_precision"] = prec
        else:
            mask_gt = y_true > thresh
            if np.any(mask_gt):
                acc = accuracy_score(y_true[mask_gt], y_pred[mask_gt])
                prec = precision_score(y_true[mask_gt], y_pred[mask_gt], average='macro', zero_division=0.0)
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
    target_ranges: dict mapping target column names to (min, max) tuples for range-based class conversion
    """

    n_samples = len(feature)
    test_split = int(n_samples * (1 - test_size))
    X_train, X_test = feature[:test_split], feature[test_split:]
    y_train, y_test = target[:test_split], target[test_split:]

    use_range_conversion = False
    target_range = None
    if target_ranges and len(target_ranges) == 1:
        target_col, (min_val, max_val) = list(target_ranges.items())[0]
        target_range = (min_val, max_val)
        use_range_conversion = True
        if verbose:
            print(f"Using range-based class conversion with range: {target_range}")
    elif target_ranges and len(target_ranges) > 1:
        if verbose:
            print("Multiple targets detected. This function should be called for each target separately.")

    if use_range_conversion:
        y_train_encoded = y_train
        y_test_encoded = y_test
        class_to_idx = None
        idx_to_class = None
    else:
        if isinstance(y_train, pd.Series):
            all_classes = sorted(y_train.unique())
        else:
            all_classes = sorted(np.unique(y_train))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        if isinstance(y_train, pd.Series):
            y_train_encoded = y_train.map(class_to_idx).values
            y_test_encoded = y_test.map(class_to_idx).values
        else:
            y_train_encoded = np.vectorize(class_to_idx.get)(y_train)
            y_test_encoded = np.vectorize(class_to_idx.get)(y_test)

    cv = TimeSeriesSplit(n_splits=k) if k > 1 else None
    metrics_dict = {}
    best_model = None

    if isinstance(param_grid, dict) and param_grid:
        from itertools import product
        keys, values = zip(*param_grid.items())
        param_dicts = [dict(zip(keys, v)) for v in product(*values)]
    else:
        param_dicts = [{}]

    for params in param_dicts:
        train_scores = []
        valid_scores = []
        train_threshold_metrics = {}
        valid_threshold_metrics = {}
        test_scores = []
        test_threshold_metrics = {}
        if cv:
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
                if verbose:
                    print(f"Training fold {fold_idx + 1}/{k} with params: {params}")
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
                model = model_wrapper_class(**params)
                if use_range_conversion:
                    model.fit(X_tr, y_tr, target_range=target_range)
                    y_val_pred_proba = model.predict_proba(X_val, target_range=target_range)
                    y_tr_pred_proba = model.predict_proba(X_tr, target_range=target_range)
                else:
                    model.fit(X_tr, y_tr)
                    y_val_pred_proba = model.predict_proba(X_val)
                    y_tr_pred_proba = model.predict_proba(X_tr)
                y_val_pred_idx = y_val_pred_proba.argmax(axis=1)
                y_tr_pred_idx = y_tr_pred_proba.argmax(axis=1)
                if use_range_conversion:
                    y_val_pred = y_val_pred_idx
                    y_val_orig = y_val
                    y_tr_pred = y_tr_pred_idx
                    y_tr_orig = y_tr
                else:
                    if idx_to_class is not None:
                        y_val_pred = np.vectorize(idx_to_class.get)(y_val_pred_idx)
                        y_val_orig = np.vectorize(idx_to_class.get)(y_val)
                        y_tr_pred = np.vectorize(idx_to_class.get)(y_tr_pred_idx)
                        y_tr_orig = np.vectorize(idx_to_class.get)(y_tr)
                    else:
                        y_val_pred = y_val_pred_idx
                        y_val_orig = y_val
                        y_tr_pred = y_tr_pred_idx
                        y_tr_orig = y_tr
                acc_valid = accuracy_score(y_val_orig, y_val_pred)
                prec_valid = precision_score(y_val_orig, y_val_pred, average='macro', zero_division=0.0)
                valid_scores.append({'accuracy': acc_valid, 'precision': prec_valid})
                acc_train = accuracy_score(y_tr_orig, y_tr_pred)
                prec_train = precision_score(y_tr_orig, y_tr_pred, average='macro', zero_division=0.0)
                train_scores.append({'accuracy': acc_train, 'precision': prec_train})
                for target_col, (min_val, max_val) in target_ranges.items():
                    th_metrics_valid = compute_threshold_metrics(y_val_orig, y_val_pred, min_val, max_val)
                    th_metrics_train = compute_threshold_metrics(y_tr_orig, y_tr_pred, min_val, max_val)
                    for k_, v_ in th_metrics_valid.items():
                        valid_threshold_metrics.setdefault(f"{target_col}_valid_{k_}", []).append(v_)
                    for k_, v_ in th_metrics_train.items():
                        train_threshold_metrics.setdefault(f"{target_col}_train_{k_}", []).append(v_)
            # After CV, fit on all training data and evaluate on test set
            model = model_wrapper_class(**params)
            if use_range_conversion:
                model.fit(X_train, y_train_encoded, target_range=target_range)
                y_test_pred_proba = model.predict_proba(X_test, target_range=target_range)
            else:
                model.fit(X_train, y_train_encoded)
                y_test_pred_proba = model.predict_proba(X_test)
            y_test_pred_idx = y_test_pred_proba.argmax(axis=1)
            if use_range_conversion:
                y_test_pred = y_test_pred_idx
                y_test_orig = y_test_encoded
            else:
                if idx_to_class is not None:
                    y_test_pred = np.vectorize(idx_to_class.get)(y_test_pred_idx)
                    y_test_orig = np.vectorize(idx_to_class.get)(y_test_encoded)
                else:
                    y_test_pred = y_test_pred_idx
                    y_test_orig = y_test_encoded
            acc_test = accuracy_score(y_test_orig, y_test_pred)
            prec_test = precision_score(y_test_orig, y_test_pred, average='macro', zero_division=0.0)
            test_scores = [{'accuracy': acc_test, 'precision': prec_test}]
            for target_col, (min_val, max_val) in target_ranges.items():
                th_metrics_test = compute_threshold_metrics(y_test_orig, y_test_pred, min_val, max_val)
                for k_, v_ in th_metrics_test.items():
                    test_threshold_metrics.setdefault(f"{target_col}_test_{k_}", []).append(v_)
        else:
            if verbose:
                print(f"Training single split with params: {params}")
            model = model_wrapper_class(**params)
            if use_range_conversion:
                model.fit(X_train, y_train_encoded, target_range=target_range)
                y_test_pred_proba = model.predict_proba(X_test, target_range=target_range)
                y_train_pred_proba = model.predict_proba(X_train, target_range=target_range)
            else:
                model.fit(X_train, y_train_encoded)
                y_test_pred_proba = model.predict_proba(X_test)
                y_train_pred_proba = model.predict_proba(X_train)
            y_test_pred_idx = y_test_pred_proba.argmax(axis=1)
            y_train_pred_idx = y_train_pred_proba.argmax(axis=1)
            if use_range_conversion:
                y_test_pred = y_test_pred_idx
                y_test_orig = y_test_encoded
                y_train_pred = y_train_pred_idx
                y_train_orig = y_train_encoded
            else:
                if idx_to_class is not None:
                    y_test_pred = np.vectorize(idx_to_class.get)(y_test_pred_idx)
                    y_test_orig = np.vectorize(idx_to_class.get)(y_test_encoded)
                    y_train_pred = np.vectorize(idx_to_class.get)(y_train_pred_idx)
                    y_train_orig = np.vectorize(idx_to_class.get)(y_train_encoded)
                else:
                    y_test_pred = y_test_pred_idx
                    y_test_orig = y_test_encoded
                    y_train_pred = y_train_pred_idx
                    y_train_orig = y_train_encoded
            acc_test = accuracy_score(y_test_orig, y_test_pred)
            prec_test = precision_score(y_test_orig, y_test_pred, average='macro', zero_division=0.0)
            acc_train = accuracy_score(y_train_orig, y_train_pred)
            prec_train = precision_score(y_train_orig, y_train_pred, average='macro', zero_division=0.0)
            valid_scores = []  # No validation in single split
            train_scores = [{'accuracy': acc_train, 'precision': prec_train}]
            test_scores = [{'accuracy': acc_test, 'precision': prec_test}]
            valid_threshold_metrics = {}
            train_threshold_metrics = {}
            test_threshold_metrics = {}
            for target_col, (min_val, max_val) in target_ranges.items():
                th_metrics_test = compute_threshold_metrics(y_test_orig, y_test_pred, min_val, max_val)
                th_metrics_train = compute_threshold_metrics(y_train_orig, y_train_pred, min_val, max_val)
                for k_, v_ in th_metrics_test.items():
                    test_threshold_metrics.setdefault(f"{target_col}_test_{k_}", []).append(v_)
                for k_, v_ in th_metrics_train.items():
                    train_threshold_metrics.setdefault(f"{target_col}_train_{k_}", []).append(v_)
        # Aggregate metrics
        avg_metrics = {}
        # Train metrics
        if train_scores:
            for k in train_scores[0]:
                avg_metrics[f"{target_col}_train_{k}"] = np.mean([score[k] for score in train_scores])
        # Validation metrics
        if valid_scores:
            for k in valid_scores[0]:
                avg_metrics[f"{target_col}_valid_{k}"] = np.mean([score[k] for score in valid_scores])
        # Test metrics (always computed)
        if test_scores:
            for k in test_scores[0]:
                avg_metrics[f"{target_col}_test_{k}"] = np.mean([score[k] for score in test_scores])
        # Add threshold metrics
        for k_, v_ in train_threshold_metrics.items():
            avg_metrics[k_] = np.mean(v_)
            avg_metrics[k_ + "_std"] = np.std(v_)
        for k_, v_ in valid_threshold_metrics.items():
            avg_metrics[k_] = np.mean(v_)
            avg_metrics[k_ + "_std"] = np.std(v_)
        for k_, v_ in test_threshold_metrics.items():
            avg_metrics[k_] = np.mean(v_)
            avg_metrics[k_ + "_std"] = np.std(v_)
        metrics_dict[str(params)] = avg_metrics
        print(avg_metrics)
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
        return_final_model=True
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
