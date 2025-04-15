# from __future__ import annotations
from gollum.featurization.deep import BaseNNFeaturizer
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from torch.optim.lr_scheduler import StepLR

from botorch.optim.fit import fit_gpytorch_mll_torch
import wandb
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

from abc import ABC, abstractmethod
from gpytorch.means.mean import Mean
from gpytorch.module import Module

from typing import Union
import numpy as np
import torch
import gpytorch
import os



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class SurrogateModel(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass



class GP(SurrogateModel, SingleTaskGP):
    def __init__(
        self,
        train_x: Union[np.ndarray, torch.Tensor] = None,
        train_y: Union[np.ndarray, torch.Tensor] = None,
        likelihood: Union[GaussianLikelihood, None] = None,
        covar_module: Union[Module, None] = None,
        mean_module: Union[Mean, None] = None,
        standardize: bool = True,
        normalize: bool = False,
        initial_noise_val: float = 1e-4,
        noise_constraint: float = 1e-5,
        initial_outputscale_val: float = 1.0,
        initial_lengthscale_val: float = 1.0,
        gp_lr: float = 0.2,
    ) -> None:

        super().__init__(
            train_X=train_x,
            train_Y=train_y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.train_x = train_x
        self.train_y = train_y

        self.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(noise_constraint)
        )

        hypers = {
            "likelihood.noise_covar.noise": torch.tensor(initial_noise_val),
            "covar_module.base_kernel.lengthscale": torch.tensor(
                initial_lengthscale_val
            ),
            "covar_module.outputscale": torch.tensor(initial_outputscale_val),
        }

        existing_parameters = {name for name, _ in self.named_parameters()}
        hypers_to_use = {
            k: torch.tensor(v)
            for k, v in hypers.items()
            if k in existing_parameters and v is not None
        }

        self.initialize(**hypers_to_use)
        self.gp_lr = gp_lr
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def fit(self):
        self.train()
        self.likelihood.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()
        mll = mll.to(self.train_x)
        

        try:
            
            fit_gpytorch_mll(
                mll
            )

        except Exception as e:
            print(f"Exception caught during fit: {str(e)}")

    def predict(
        self, x, observation_noise=False, return_var=True, return_posterior=False
    ):
        self.eval()
        self.likelihood.eval()

        with torch.no_grad():
            posterior = self.posterior(x, observation_noise=observation_noise)
        return (
            posterior
            if return_posterior
            else (posterior.mean, posterior.variance) if return_var else posterior.mean
        )


class DeepGP(SurrogateModel, SingleTaskGP):
    def __init__(
        self,
        train_x: Union[np.ndarray, torch.Tensor] = None,
        train_y: Union[np.ndarray, torch.Tensor] = None,
        likelihood: Union[GaussianLikelihood, None] = None,
        covar_module: Union[Module, None] = None,
        mean_module: Union[Mean, None] = None,
        standardize: bool = True,
        normalize: bool = False,
        initial_noise_val: float = 1e-4,
        noise_constraint: float = 1e-5,
        initial_outputscale_val: float = 2.0,
        initial_lengthscale_val: float = 5,
        ft_lr: float = 0.002,
        gp_lr: float = 0.02,
        gp_step_lr: float = 0.95,
        wd: float = 1e-3,
        wd_llm: float = 1e-3,
        scale_embeddings: bool = False,
        train_mll_additionally: bool = False,
        finetuning_model: Union[None, BaseNNFeaturizer] = None,
    ) -> None:

        tkwargs = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64,
        }
        train_x = train_x.to(torch.float64)
        train_y = train_y.to(**tkwargs)

        # TODO device handling
        super().__init__(
            train_X=train_x,
            train_Y=train_y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=Standardize(train_y.shape[-1]) if standardize else None,
            input_transform=Normalize(train_x.shape[-1]) if normalize else None,
        )

        self.train_x = train_x
        self.train_y = train_y

        self.mean_module = mean_module
        self.covar_module = covar_module

        self.finetuning_model = finetuning_model
        self.finetuning_model = self.finetuning_model.to(**tkwargs)
        self.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(noise_constraint)
        )

        hypers = {
            "likelihood.noise_covar.noise": torch.tensor(initial_noise_val),
            "covar_module.base_kernel.lengthscale": torch.tensor(
                initial_lengthscale_val
            ),
            "covar_module.outputscale": torch.tensor(initial_outputscale_val),
        }

        existing_parameters = {name for name, _ in self.named_parameters()}
        hypers_to_use = {
            k: torch.tensor(v)
            for k, v in hypers.items()
            if k in existing_parameters and v is not None
        }
        self.initialize(**hypers_to_use)
        self.train_x = self.train_x.to(**tkwargs)
        self.train_y = self.train_y.to(**tkwargs)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        self.ft_lr = ft_lr
        self.gp_lr = gp_lr
        self.gp_step_lr = gp_step_lr
        self.wd = wd
        self.wd_llm = wd_llm
        self.scale_embeddings = scale_embeddings
        self.train_mll_additionally = train_mll_additionally

        self.to_gpu()

    def forward(self, x):
        finetuned = self.finetuning_model(x)

        if self.scale_embeddings:
            finetuned = self.scale_to_bounds(finetuned)
        self.finetuned = finetuned

        mean_x = self.mean_module(self.finetuned)
        covar_x = self.covar_module(self.finetuned)
        
        wandb.log({"lr/llm_lr": self.optimizer.param_groups[0]["lr"]})
        wandb.log({"lr/gp_lr": self.optimizer.param_groups[1]["lr"]})

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def to_gpu(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.likelihood.to(device)
        self.finetuning_model.to(device)
        self.train_x = self.train_x.to(device)
        self.train_targets = self.train_targets.to(device)
        self.train_y = self.train_y.to(device)

    def fit(self):
        self.train()
        self.likelihood.train()
        self.finetuning_model.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        mll.train()
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mll.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        def gp_closure():
            self.optimizer.zero_grad()
            output = self(self.train_x)
            mll_loss = -mll(output, self.train_targets.squeeze())
            mll_loss.backward()
            grads = [p.grad for p in self.parameters() if p.requires_grad]
            return mll_loss, grads

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": (
                        p for p in self.finetuning_model.parameters() if p.requires_grad
                    ),
                    "lr": self.ft_lr,
                    "weight_decay": self.wd_llm,
                },
                {"params": self.covar_module.parameters()},
                {"params": self.mean_module.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.gp_lr,
            weight_decay=self.wd,
        )
        
        scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"Total number of parameters: {total_params}")
        
        
        fit_gpytorch_mll(
            mll,
            closure=gp_closure,
            optimizer=fit_gpytorch_mll_torch,
            optimizer_kwargs={"optimizer": self.optimizer, "scheduler": scheduler},
            
        )

        if self.train_mll_additionally:
            for param in self.finetuning_model.parameters():
                param.requires_grad = False
            fit_gpytorch_mll(mll)

        

    def predict(
        self, x, observation_noise=True, return_var=True, return_posterior=False
    ):
        torch.cuda.empty_cache()
        for param in self.finetuning_model.parameters():
            param.requires_grad = False

        self.eval()
        self.finetuning_model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            posterior = self.posterior(x, observation_noise=observation_noise)

        return (
            posterior
            if return_posterior
            else (posterior.mean, posterior.variance) if return_var else posterior.mean
        )


