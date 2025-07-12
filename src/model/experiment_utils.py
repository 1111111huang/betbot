import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import accuracy_score, precision_score
from src.model.training import train_model_and_collect_metrics
from src.model.experiment_utils import *
from itertools import product
import mlflow
from itertools import product

def plot_metrics_for_target_and_model(
    experiment_name, target_col, model_name, tracking_uri, top_n=5
):
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Search runs
    filter_string = f"params.target = '{target_col}' and params.model_name = '{model_name}'"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        output_format="pandas"
    )

    if runs.empty:
        print("No matching runs found.")
        return

    # Extract test metrics without standard deviations
    metric_rows = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_name = run.get("tags.mlflow.runName", run_id)
        metrics = {}
        
        # Look for all test metrics in the run using exact metric names
        for k, v in run.items():
            if (f"{target_col}_test_") in k and \
                ("_accuracy" in k or "_precision" in k) and \
                not k.endswith("_std"):
                metrics[k] = v
        
        if metrics:
            metric_rows.append({
                "run_name": run_name,
                "run_id": run_id,
                "metrics": metrics
            })

    if not metric_rows:
        print("No test metrics found.")
        return

    # Calculate average test metrics for ranking
    avg_scores = []
    for row in metric_rows:
        metrics = row["metrics"]
        avg_acc = np.mean([v for k, v in metrics.items() if "accuracy" in k])
        avg_prec = np.mean([v for k, v in metrics.items() if "precision" in k])
        avg_scores.append({
            "run_id": row["run_id"],
            "run_name": row["run_name"],
            "avg_test_accuracy": avg_acc,
            "avg_test_precision": avg_prec
        })
    
    avg_scores_df = pd.DataFrame(avg_scores)
    top_acc_runs = avg_scores_df.nlargest(top_n, "avg_test_accuracy")
    top_prec_runs = avg_scores_df.nlargest(top_n, "avg_test_precision")

    # Create figure with subplots vertically arranged
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
    
    for ax, metric_type, top_runs in [
        (ax1, "accuracy", top_acc_runs),
        (ax2, "precision", top_prec_runs)
    ]:
        for _, run in top_runs.iterrows():
            run_data = next(r for r in metric_rows if r["run_id"] == run["run_id"])
            metrics = run_data["metrics"]
            
            # Collect metrics by type
            x_labels = []
            values = []
            
            # First add lte metrics
            lte_metrics = sorted(
                [(k, v) for k, v in metrics.items() 
                 if f"{target_col}_test_lte_" in k and f"_{metric_type}" in k],
                key=lambda x: int(x[0].split("_")[-2])
            )
            for k, v in lte_metrics:
                x_labels.append(f"lte_{k.split('_')[-2]}")
                values.append(v)
            
            # Then add gt metrics
            gt_metrics = sorted(
                [(k, v) for k, v in metrics.items() 
                 if f"{target_col}_test_gt_" in k and f"_{metric_type}" in k],
                key=lambda x: int(x[0].split("_")[-2])
            )
            for k, v in gt_metrics:
                x_labels.append(f"gt_{k.split('_')[-2]}")
                values.append(v)
            
            # Finally add top1 if present
            top1_key = f"{target_col}_test_top1_{metric_type}"
            if top1_key in metrics:
                x_labels.append("top1")
                values.append(metrics[top1_key])
            
            ax.plot(x_labels, values, marker='o', label=run_data["run_name"],
                   markersize=8, linewidth=2)
        
        ax.set_title(f'Test {metric_type.capitalize()} Metrics (Top {top_n} Models)')
        ax.set_xlabel('Metric Type')
        ax.set_ylabel(f'Test {metric_type.capitalize()}')
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

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

    # Sort by date if it's in key_columns
    if 'date' in key_columns:
        merged['date'] = pd.to_datetime(merged['date'])
        merged = merged.sort_values('date')

    # Separate aligned outputs
    X_aligned = merged[feature_df.columns]
    target_aligned = merged[target_df.columns]

    return X_aligned.reset_index(drop=True), target_aligned.reset_index(drop=True)

def plot_run_metrics_by_split(experiment_name, run_name, target_col, tracking_uri):
    """
    Plot train, valid, and test metrics for a given run in an MLflow experiment.
    Metrics are already averaged across folds.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    # Find the run by name
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        output_format="pandas"
    )
    if runs.empty:
        print(f"No run found with name '{run_name}'.")
        return

    run = runs.iloc[0]
    
    # Create figure with subplots vertically arranged
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
    
    for ax, metric_type in [(ax1, "accuracy"), (ax2, "precision")]:
        splits = ["train", "valid", "test"]
        x_labels = None
        
        for split in splits:
            metrics = {}
            # Collect all metrics for this split and metric type (excluding std)
            split_metrics = {
                k: v for k, v in run.items() 
                if (f"{target_col}_{split}_" in k) and 
                k.endswith(f"_{metric_type}") and 
                not k.endswith("_std")
            }
            
            if not x_labels:
                # Create x_labels based on metric types in a specific order
                lte_metrics = sorted(
                    [(k, v) for k, v in split_metrics.items() if "_lte_" in k],
                    key=lambda x: int(x[0].split("_")[-2])
                )
                gt_metrics = sorted(
                    [(k, v) for k, v in split_metrics.items() if "_gt_" in k],
                    key=lambda x: int(x[0].split("_")[-2])
                )
                
                x_labels = (
                    [f"lte_{k.split('_')[-2]}" for k, _ in lte_metrics] +
                    [f"gt_{k.split('_')[-2]}" for k, _ in gt_metrics]
                )
                print(lte_metrics, gt_metrics)
                print(x_labels)
            
            values = []
            for label in x_labels:
                if label.startswith("lte_"):
                    threshold = label.split("_")[1]
                    key = f"metrics.{target_col}_{split}_lte_{threshold}_{metric_type}"
                else:  # gt
                    threshold = label.split("_")[1]
                    key = f"metrics.{target_col}_{split}_gt_{threshold}_{metric_type}"
                values.append(run[key])
            
            ax.plot(x_labels, values, marker='o', label=split,
                   markersize=8, linewidth=2)
        
        ax.set_title(f'{metric_type.capitalize()} Metrics Across Splits')
        ax.set_xlabel('Metric Type')
        ax.set_ylabel(f'{metric_type.capitalize()}')
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

def run_multiclass_distribution_experiment(
    feature_df,
    target_df,
    target_ranges,
    model_wrapper_class,
    model_param_grid,
    test_size,
    experiment_name,
    uri,
    model_name,
    k,
    key_columns,
):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    results = {}
    for target_col, (min_val, max_val) in target_ranges.items():
        feature_columns = [c for c in feature_df.columns if c not in key_columns]
        feature = feature_df[feature_columns].values
        target = target_df[target_col].values

        print(f"Training {model_name} on {target_col} with param grid: {model_param_grid}")
        param_metrics, _ = train_model_and_collect_metrics(
            feature=feature,
            target=target,
            target_ranges={target_col: (min_val, max_val)},
            model_wrapper_class=model_wrapper_class,
            param_grid=model_param_grid,
            k=k,
            key_columns=key_columns,
            test_size=test_size,
            return_final_model=False,
            include_key_columns=False,
            verbose=True
        )
        # Log metrics to MLflow
        for param_set, metrics in param_metrics.items():
            with mlflow.start_run(run_name=f"{model_name}_{target_col}_{param_set}"):
                if isinstance(param_set, dict):
                    mlflow.log_params(param_set)
                else:
                    mlflow.log_param("param_set", param_set)
                mlflow.log_param("target_col", target_col)
                for metric_name, metric_val in metrics.items():
                    mlflow.log_metric(metric_name, metric_val)
            results[(target_col, str(param_set))] = metrics
    return results

def plot_run_metrics_side_by_side(experiment_name, model_name, model_param, tracking_uri, metric_type="accuracy"):
    """
    Plot train, valid, and test metrics side by side for all runs matching model_name and model_param.
    Each subplot is a run (target variable), showing metric from lte to gt in increasing order.
    Title for each graph is the target variable name from the run_name.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    # Build filter string for model_name and all param values
    filter_clauses = [f"params.model_name = '{model_name}'"]
    for k, v in model_param.items():
        filter_clauses.append(f"params.{k} = '{v}'")
    filter_string = " and ".join(filter_clauses)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        output_format="pandas"
    )
    if runs.empty:
        print(f"No runs found with model_name '{model_name}' and params {model_param}.")
        return

    # For each run, plot a line graph of the metric from lte to gt in increasing order
    n_runs = len(runs)
    n_cols = min(3, n_runs)
    n_rows = (n_runs + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))

    for idx, (_, run) in enumerate(runs.iterrows(), 1):
        ax = plt.subplot(n_rows, n_cols, idx)
        target_col = run["params.target"]
        run_name = run.get("tags.mlflow.runName", "")
        splits = ["train", "valid", "test"]

        # Collect all lte/gt metrics for the test split, sorted by threshold
        test_metrics = {
            k: v for k, v in run.items()
            if (f"{target_col}_test_" in k) and
               (k.endswith(f"_{metric_type}")) and
               not k.endswith("_std")
        }
        # lte metrics
        lte_metrics = sorted(
            [(k, v) for k, v in test_metrics.items() if "_lte_" in k],
            key=lambda x: int(x[0].split("_")[-2])
        )
        # gt metrics
        gt_metrics = sorted(
            [(k, v) for k, v in test_metrics.items() if "_gt_" in k],
            key=lambda x: int(x[0].split("_")[-2])
        )
        # Compose x_labels and values
        x_labels = [f"lte_{k.split('_')[-2]}" for k, _ in lte_metrics] + [f"gt_{k.split('_')[-2]}" for k, _ in gt_metrics]
        values = [v for _, v in lte_metrics] + [v for _, v in gt_metrics]

        ax.plot(x_labels, values, marker='o', linewidth=2)
        ax.set_title(target_col)
        ax.set_xlabel('Metric Type')
        ax.set_ylabel(metric_type.capitalize())
        ax.set_ylim(0, 1)
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.suptitle(f'{metric_type.capitalize()} Metrics by Threshold for model {model_name}\nParams: {model_param}', y=1.02)
    plt.tight_layout()
    plt.show()

def make_sequence_dataset(feature_df, target_df, key_columns, sequence_length=5, target_col=None, pad_value=0):
    """
    Convert flat feature/target DataFrames into sequences for temporal models, with padding for short histories.
    Returns: X_seq [N, T, F], y_seq [N]
    """
    feature_columns = [c for c in feature_df.columns if c not in key_columns]
    all_sequences = []
    all_targets = []
    group_key = [k for k in key_columns if k != 'date']
    # Merge feature and target for grouping
    merged = feature_df.copy()
    if target_col is not None and target_col in target_df.columns:
        merged[target_col] = target_df[target_col]
    grouped = merged.groupby(group_key)
    for _, group in grouped:
        group = group.sort_values('date')
        features = group[feature_columns].values
        targets = group[target_col].values
        n = len(group)
        for i in range(n):
            # For each row, build a sequence ending at i (inclusive), pad if needed
            start_idx = max(0, i - sequence_length)
            seq = features[start_idx:i]
            # Pad at the beginning if needed
            if seq.shape[0] < sequence_length:
                pad_width = sequence_length - seq.shape[0]
                pad = np.full((pad_width, features.shape[1]), pad_value, dtype=features.dtype)
                seq = np.vstack([pad, seq])
            all_sequences.append(seq)
            all_targets.append(targets[i])
    if not all_sequences:
        raise ValueError("No sequences generated. Check your group_key, sequence_length, and data size.")
    return np.stack(all_sequences), np.array(all_targets)

def run_multiclass_distribution_experiment_seq(
    feature_df,
    target_df,
    target_ranges,
    model_wrapper_class,
    model_param_grid,
    test_size,
    experiment_name,
    uri,
    model_name,
    k,
    key_columns,
    sequence_length=5,
    pad_value=0,
):
    mlflow.set_tracking_uri(uri)
    results = {}
    for target_col, (min_val, max_val) in target_ranges.items():
        print(f"Preparing sequence data for target: {target_col}")
        X_seq, y_seq = make_sequence_dataset(
            feature_df, target_df, key_columns,
            sequence_length=sequence_length,
            target_col=target_col,
            pad_value=pad_value
        )
        print(f"  X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

        print(f"Training {model_name} on {target_col} with param grid: {model_param_grid}")
        param_metrics, _ = train_model_and_collect_metrics(
            feature=X_seq,
            target=y_seq,
            target_ranges={target_col: (min_val, max_val)},
            model_wrapper_class=model_wrapper_class,
            param_grid=model_param_grid,
            k=k,
            key_columns=key_columns,
            test_size=test_size,
            return_final_model=False,
            include_key_columns=False,
            verbose=True
        )
        # Log metrics to MLflow
        for param_set, metrics in param_metrics.items():
            with mlflow.start_run(run_name=f"{model_name}_{target_col}_{param_set}"):
                if isinstance(param_set, dict):
                    mlflow.log_params(param_set)
                else:
                    mlflow.log_param("param_set", param_set)
                mlflow.log_param("target_col", target_col)
                for metric_name, metric_val in metrics.items():
                    mlflow.log_metric(metric_name, metric_val)
            results[(target_col, str(param_set))] = metrics
    return results