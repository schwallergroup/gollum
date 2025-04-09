from gpytorch.metrics import (
    mean_standardized_log_loss,
    negative_log_predictive_density,
    quantile_coverage_error,
)
import wandb
import torch
import torch.nn.functional as F


def log_model_parameters(model, epoch=0):
    for name, param in model.named_hyperparameters():
        if "finetuning_model" not in name:
            transformed_name = name.replace("raw_", "")
            attr = model
            for part in transformed_name.split("."):
                attr = getattr(attr, part)
            if isinstance(attr, torch.Tensor):
                value = attr.cpu().detach().numpy()
            else:
                value = attr

        wandb.log({transformed_name: value, "epoch": epoch})


def calculate_model_fit_metrics(posterior, y, stage="train"):
    preds, variances = posterior.mean, posterior.variance

    preds = posterior.mean.squeeze()
    variances = posterior.variance.squeeze()
    y = y.squeeze()
    mse = F.mse_loss(preds, y)
    mae = torch.mean(torch.abs(preds - y))
    r2 = 1 - (torch.sum((y - preds) ** 2) / torch.sum((y - torch.mean(y)) ** 2))
    nlpd = negative_log_predictive_density(posterior, y)
    msll = mean_standardized_log_loss(posterior, y)
    qce = quantile_coverage_error(posterior, y)

    metrics = {
        f"{stage}/mse": mse.item(),
        f"{stage}/mae": mae.item(),
        f"{stage}/r2": r2.item(),
        f"{stage}/nlpd": nlpd.item(),
        f"{stage}/msll": msll.item(),
        f"{stage}/qce": qce.item(),
    }
    
   
    return metrics

def calculate_weighted_metrics(preds, y, stage="full"):
    quantiles = [0.99, 0.95, 0.90, 0.75]
    weighted_metrics = {}

    for q in quantiles:
        threshold = torch.quantile(y, q)
        weights = torch.where(y >= threshold, 3.0, 1.0)

        w_mse = torch.sum(weights * (preds - y)**2) / torch.sum(weights)
        w_mae = torch.sum(weights * torch.abs(preds - y)) / torch.sum(weights)
        w_r2 = 1 - (torch.sum(weights * (y - preds) ** 2) /
                    torch.sum(weights * (y - torch.mean(y)) ** 2))

        weighted_metrics[f"{stage}/weighted_mse@{int(q*100)}"] = w_mse.item()
        weighted_metrics[f"{stage}/weighted_mae@{int(q*100)}"] = w_mae.item()
        weighted_metrics[f"{stage}/weighted_r2@{int(q*100)}"] = w_r2.item()
    
    r2 = 1 - (torch.sum((y - preds) ** 2) / torch.sum((y - torch.mean(y)) ** 2))
    mse = F.mse_loss(preds, y)
    mae = torch.mean(torch.abs(preds - y))
    
    weighted_metrics[f"{stage}/mae"] = mae.item()
    weighted_metrics[f"{stage}/mse"] = mse.item()
    weighted_metrics[f"{stage}/r2"] = r2.item()

    return weighted_metrics


def log_model_fit_metrics(metrics, epoch=0, log_images=False):
    """
    Log model fit metrics to WandB.
    """
    log_data = {key: value for key, value in metrics.items()}
    log_data["epoch"] = epoch
    wandb.log(log_data)
