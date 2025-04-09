import pandas as pd
from typing import Dict, Optional
import numpy as np
import torch
from pytorch_metric_learning import distances as dist
from bochemian.surrogate_models.gp import SurrogateModel


def compute_cumulative_metrics(
    group: pd.DataFrame, theoretical_max: float = 100.0
) -> Dict[str, float]:
    """
    Compute cumulative metrics for a group of runs.
    """
    metrics = {}

    # Compute area under the best-so-far curve (higher is better)
    metrics["cumulative_best"] = np.trapz(group["train/best_so_far"])

    max_value = group["train/best_so_far"].max()
    theoretical_max_area = max_value * len(group["train/best_so_far"])
    metrics["normalized_cumulative_best"] = (
        metrics["cumulative_best"] / theoretical_max_area
    )

    # Compute cumulative counts (higher is better)
    bo_metrics_final_epoch = group[group["epoch"] == group["epoch"].max()]
    for count_col in [
        "quantile_99_count",
        "quantile_95_count",
        "quantile_90_count",
        "quantile_75_count",
    ]:
        if count_col in group.columns:
            metrics[f"cumulative_{count_col}"] = group[count_col].sum()
            metrics[f"final_{count_col}"] = bo_metrics_final_epoch[count_col].mean()

    # Compute mean final metrics
    model_metrics_final_epoch = group[group["epoch"] == group["epoch"].max() - 1]
    for metric in [
        "train/nlpd",
        "train/mse",
        "train/r2",
        "train/msll",
        "train/qce",
        "valid/nlpd",
        "valid/mse",
        "valid/r2",
        "valid/msll",
        "valid/qce",
        "covar_module.base_kernel.lengthscale",
        "covar_module.outputscale",
        "likelihood.noise_covar.noise",
    ]:
        metrics[f"final_{metric}"] = model_metrics_final_epoch[metric].mean()

    return metrics


def analyze_results(
    df: pd.DataFrame, param_cols=None, sort_by_column="cumulative_best"
) -> pd.DataFrame:
    """
    Analyze sweep results and return a summary dataframe sorted by performance.
    """

    # Compute metrics for each parameter configuration
    results = []
    for params, group in df.groupby(param_cols, dropna=False):
        if len(param_cols) == 1:
            param_dict = {param_cols[0]: params}
        else:
            param_dict = dict(zip(param_cols, params))

        # Compute metrics across all seeds
        seed_metrics = []
        for seed, seed_group in group.groupby("seed"):
            metrics = compute_cumulative_metrics(
                seed_group, theoretical_max=seed_group["summary_target_stat_max"].max()
            )
            seed_metrics.append(metrics)

        # Average metrics across seeds
        avg_metrics = {
            k: np.mean([m[k] for m in seed_metrics]) for k in seed_metrics[0].keys()
        }
        std_metrics = {
            f"{k}_std": np.std([m[k] for m in seed_metrics])
            for k in seed_metrics[0].keys()
        }
        # TODO save only one
        count_metrics = {
            f"{k}_cnt": len([m[k] for m in seed_metrics])
            for k in seed_metrics[0].keys()
        }

        results.append({**param_dict, **avg_metrics, **std_metrics, **count_metrics})

    results_df = pd.DataFrame(results)

    # Sort by cumulative best-so-far (can be changed to other metrics)
    results_df = results_df.sort_values(sort_by_column, ascending=False)

    return results_df


def compute_thresholds(y, low_quantile=0.2, high_quantile=0.8):
    """
    Computes thresholds using 5th and 95th percentiles.

    Args:
        y: Objective values
        low_quantile: Quantile for low threshold (default: 0.05 for 5th percentile)
        high_quantile: Quantile for high threshold (default: 0.95 for 95th percentile)
    """
    if torch.is_tensor(y):
        low_threshold = torch.quantile(y, low_quantile)
        high_threshold = torch.quantile(y, high_quantile)
    else:
        low_threshold = np.quantile(y, low_quantile)
        high_threshold = np.quantile(y, high_quantile)

    return low_threshold, high_threshold


def calculate_distances(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    high_score_threshold: float = 70,
    low_score_threshold: float = 10,
    model: Optional[SurrogateModel] = None,
):
    """
    Calculates distances between points in latent space based on their score categories.

    Args:
        embeddings: Embeddings in latent space (torch.Tensor)
        scores: Corresponding scores (torch.Tensor)
        high_score_threshold: Threshold for high scores (default: 70)
        low_score_threshold: Threshold for low scores (default: 10)
        model: Optional surrogate model for kernel similarity

    Returns:
        Tuple of (high-high distances, high-low distances, low-low distances, average distance)
    """
    # convert numpy
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)

    # kernel similrity
    if model:
        with torch.no_grad():
            distances = model.covar_module(
                embeddings.to(model.covar_module.device)
            ).evaluate()
    else:
        # Calculate pairwise L2 distances
        distances = torch.cdist(embeddings, embeddings, p=2)

    # Get masks for high and low scores
    high_mask = scores >= high_score_threshold
    low_mask = scores < low_score_threshold

    # Calculate distances for each category
    hh_distances = distances[high_mask][:, high_mask].flatten()
    hl_distances = distances[high_mask][:, low_mask].flatten()
    ll_distances = distances[low_mask][:, low_mask].flatten()
    avg_distance = distances.mean()

    return hh_distances, hl_distances, ll_distances, avg_distance
