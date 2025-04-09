import torch
import wandb


def calculate_data_stats(x, y):
    """
    Calculate statistics of the dataset.
    """
    stats = {
        "target_stat_max": torch.max(y.float()),
        "target_stat_mean": torch.mean(y.float()),
        "target_stat_std": torch.std(y.float()),
        "target_stat_var": torch.var(y.float()),
        "input_stat_feature_dimension": x.shape[-1],
        "input_stat_n_points": x.shape[0],
    }
    # Adding quantiles and top values
    for q in [0.75, 0.9, 0.95, 0.99]:
        stats[f"target_q{int(q * 100)}"] = torch.quantile(y.float(), q)
    for n in [1, 3, 5, 10]:
        top_values, _ = torch.topk(y, n, dim=0)
        stats[f"top_{n}"] = top_values[-1]
    return stats


def calculate_data_stats_numpy(x_np, y_np):
    """
    Wrapper for calculate_data_stats that accepts numpy arrays.

    Args:
        x_np (np.ndarray): Input features as numpy array
        y_np (np.ndarray): Target values as numpy array
    """
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    return calculate_data_stats(x, y)


def log_data_stats(data_metrics):
    """
    Log data statistics to WandB summary.
    """
    for key, value in data_metrics.items():
        wandb.summary[key] = value.item() if torch.is_tensor(value) else value


def log_bo_metrics(data_stats, train_y, epoch=0):
    """
    Log bo-specific metrics (quantiles, top counts and best so far) to WandB.
    """
    log_best_so_far(train_y, epoch)
    log_top_n_counts(data_stats, train_y, epoch)
    log_quantile_counts(data_stats, train_y, epoch)


def log_best_so_far(train_y, epoch=0):
    """
    Log the best-so-far value to WandB.
    """
    best_so_far = torch.max(train_y).item()
    wandb.log({"train/best_so_far": best_so_far, "epoch": epoch})


def log_top_n_counts(data_stats, train_y, epoch=0):
    """
    Log the count of top N values to WandB.
    """
    for n in [1, 3, 5, 10]:
        threshold = data_stats[f"top_{n}"]
        count = (train_y >= threshold).sum().item()
        wandb.log({f"top_{n}_count": count, "epoch": epoch})


def log_quantile_counts(data_stats, train_y, epoch=0):
    """
    Log the count of quantiles to WandB.
    """
    for q in [0.75, 0.9, 0.95, 0.99]:
        threshold = data_stats[f"target_q{int(q * 100)}"]
        count = (train_y >= threshold).sum().item()
        wandb.log({f"quantile_{int(q * 100)}_count": count, "epoch": epoch})
