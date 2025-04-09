import warnings
from botorch.exceptions import InputDataWarning

import re
import torch
import logging

warnings.filterwarnings("ignore", category=InputDataWarning)
logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
warnings.filterwarnings(
    "ignore",
    message="ExpectedImprovement has known numerical issues that lead to suboptimal optimization performance"
)

class IgnoreDeviceFilter(logging.Filter):
    def filter(self, record):
        return "available:" not in record.getMessage()


logger.addFilter(IgnoreDeviceFilter())


warnings.filterwarnings(
    "ignore",
    message=re.escape(
        "You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. "
        "To properly utilize them, you should set "
        "`torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision "
        "for performance. For more details, read "
        "https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html"
    ),
    category=UserWarning,
    module="torch",
)

warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers which may be a bottleneck.*",
    category=UserWarning,
    module="pytorch_lightning.trainer.connectors.data_connector",
)

from gollum.data.module import BaseDataModule
from gollum.bo.optimizer import BotorchOptimizer


from gollum.metrics import (
    calculate_data_stats,
    log_bo_metrics,
    log_data_stats,
)



torch.set_float32_matmul_precision("high")


from pytorch_lightning import seed_everything
import wandb
from tqdm import tqdm
from gollum.utils.config import flatten

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
)
from gollum.utils.config import instantiate_class
from botorch.acquisition import AcquisitionFunction
from gollum.surrogate_models.gp import SurrogateModel



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_EMBEDDING_SIZES = {
    "WhereIsAI/UAE-Large-V1": 1024,
    "nomic-ai/modernbert-embed-base": 768,
    "Qwen/Qwen2-7B-Instruct": 3584,
    "t5-base": 768,
    "mistralai/Mistral-7B-Instruct-v0.2": 4096,
    "text-embedding-3-large": 3072,
    "nomic-ai/modernbert-embed-base;get_huggingface_embeddings;normalize:False;pooling:cls": 768,
    "Qwen/Qwen2-7B-Instruct;get_huggingface_embeddings;normalize:False;pooling:last_token": 3584,
    "GT4SD/multitask-text-and-chemistry-t5-base-augm": 768, 
}


def configure_embedding_size(config, model_name):
    if model_name in MODEL_EMBEDDING_SIZES:
        config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"][
            "input_dim"
        ] = MODEL_EMBEDDING_SIZES[model_name]
    else:
        raise ValueError(f"Model {model_name} not found in supported models.")
    return config


def configure_pooling_method(config, model_name):
    hugging_face_models = {
        "nomic-ai/modernbert-embed-base": "cls",
        "mixedbread-ai/mxbai-embed-large-v1": "cls",
        "WhereIsAI/UAE-Large-V1": "cls",
        "Alibaba-NLP/gte-Qwen1.5-7B-instruct": "last_token_pool",
        "GT4SD/multitask-text-and-chemistry-t5-base-augm": "average",
        "GT4SD/multitask-text-and-chemistry-t5-base-augm-from-rxn": "average",
        "t5-base": "average",
        "Qwen/Qwen2-7B-Instruct": "last_token_pool",
      
    }

    if model_name in hugging_face_models:
        config["data"]["init_args"]["featurizer"]["init_args"]["pooling_method"] = (
            hugging_face_models[model_name]
        )
    else:
        raise ValueError(
            f"Model {model_name} not found in supported models. Please specify pooling method manually."
        )

    if (
        config["surrogate_model"]["class_path"]
        in ["gollum.surrogate_models.gp.DeepGP"]
    ):
        config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"][
            "pooling_method"
        ] = hugging_face_models[model_name]

    return config


def configure_benchmark_datasets(config):
    """Configure dataset settings based on benchmark name."""
    benchmark = config["benchmark"]
    if benchmark.startswith("bh"):
        reaction_num = benchmark[-1]

        config["data"]["init_args"][
            "data_path"
        ] = f"data/reactions/buchwald-hartwig/bh_reaction_{reaction_num}_procedure_template_basic.csv"
        
        config["data"]["init_args"]["target_column"] = "objective"
        config["data"]["init_args"]["maximize"] = True
    

    return config


def validate_configuration(config):
    """Validate that the configuration is consistent."""
    surrogate_class = config["surrogate_model"]["class_path"]
    
    featurizer_config = config["data"]["init_args"]["featurizer"]["init_args"]
    representation = featurizer_config.get("representation")
    
    # check for invalid configurations
    if surrogate_class == "gollum.surrogate_models.gp.GP" and representation == "get_tokens":
        raise ValueError("Standard GP or PLLM shouldn't use 'get_tokens'. This is for trainable LLM models only.")
    
    # Ensure model embedding sizes are correct
    model_name = featurizer_config.get("model_name")
    if model_name in MODEL_EMBEDDING_SIZES:
        embedding_size = MODEL_EMBEDDING_SIZES[model_name]
        
        if "surrogate_model" in config and "init_args" in config["surrogate_model"]:
            if "finetuning_model" in config["surrogate_model"]["init_args"]:
                current_dim = config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"].get("input_dim")
                if current_dim != embedding_size:
                    print(f"Updating input_dim from {current_dim} to {embedding_size} for {model_name}")
                    config["surrogate_model"]["init_args"]["finetuning_model"]["init_args"]["input_dim"] = embedding_size
    
    return config


def setup_data(config):
    initializer = instantiate_class(
        config["data"]["init_args"]["initializer"], seed=config["seed"]
    )
    featurizer = instantiate_class(config["data"]["init_args"]["featurizer"])
    dm = instantiate_class(
        config["data"],
        initializer=initializer,
        featurizer=featurizer,
        normalize_input=config["data"]["init_args"]["normalize_input"],
        maximize=config["data"]["init_args"]["maximize"],
    )

    return dm




def setup_bo_optimizer(config, design_space):
    bo_config = config["bo"]["init_args"]
    surrogate_model_config = config["surrogate_model"]
    acquisition_config = config["acquisition"]
    bo = BotorchOptimizer(
        design_space=design_space,
        surrogate_model_config=surrogate_model_config,
        acq_function_config=acquisition_config,
        batch_strategy=bo_config["batch_strategy"],
        batch_size=bo_config["batch_size"],
    )
    return bo


def train(config):
    if config.get("benchmark", None) is not None:
        config = configure_benchmark_datasets(config)
    
    config = validate_configuration(config)
    wandb_config = flatten(config)
    
    with wandb.init(
        project="bochemian_paper", config=wandb_config, group=config["group"]
    ) as run:

        dm = setup_data(config)
        bo = setup_bo_optimizer(config, design_space=dm.heldout_x)
        
        data_stats = calculate_data_stats(dm.x, dm.y)
        log_data_stats(data_stats)

        # Start the training loop
        for i in tqdm(range(config["n_iters"]), colour="blue"):
            train_x = dm.train_x.clone().to("cuda")
            train_y = dm.train_y.clone().to("cuda")
            design_space = dm.heldout_x.clone().to("cuda")

            ## this trains the model, updates acqf and returns the next point to evaluate
            x_next = bo.suggest_next_experiments(train_x, train_y, design_space)
            x_next = torch.stack(x_next)

            log_bo_metrics(data_stats, dm.train_y, epoch=i)
           

            matches = (design_space.unsqueeze(0).to("cuda") == x_next).all(dim=-1)
            indices = matches.nonzero(as_tuple=True)[1].to("cpu")

            if not torch.all(matches.sum(dim=-1) == 1):
                print("Unable to find a unique match for some x_next in the dataset.")

            wandb.log(
                {
                    "evaluated_suggestions": wandb.Histogram(dm.heldout_y[indices]),
                    "epoch": i,
                }
            )

            x_next = x_next.squeeze(1)

            # update indices tracking
            evaluated_original_indices = dm.heldout_indices[indices]
            dm.train_indexes = np.append(dm.train_indexes, evaluated_original_indices)
            dm.heldout_indices = np.delete(dm.heldout_indices, indices)

            dm.train_x = dm.x[dm.train_indexes]
            dm.train_y = dm.y[dm.train_indexes]
            dm.heldout_x = dm.x[dm.heldout_indices]
            dm.heldout_y = dm.y[dm.heldout_indices]

            train_df = dm.data.loc[dm.train_indexes].copy()
            heldout_df = dm.data.loc[dm.heldout_indices.tolist()].copy()
            dm.data = pd.concat([train_df, heldout_df])

            df_values = dm.data.loc[dm.train_indexes][dm.target_column].values
            tensor_values = dm.train_y.squeeze().cpu().numpy()

            is_consistent = np.allclose(df_values, tensor_values)
            assert is_consistent, "DataFrame values don't match tensor values"

            assert len(np.unique(dm.train_indexes)) == len(
                dm.train_indexes
            ), "Duplicates found in dm.train_indexes"
            assert len(np.unique(dm.heldout_indices)) == len(
                dm.heldout_indices
            ), "Duplicates found in dm.heldout_indices"

            # Check for any common indices between dm.train_indexes and dm.heldout_indices
            common_indices = np.intersect1d(dm.train_indexes, dm.heldout_indices)
            assert (
                len(common_indices) == 0
            ), f"Common indices found between train and heldout: {common_indices}"
            total_indices = len(dm.train_indexes) + len(dm.heldout_indices)
            assert total_indices == len(dm.x), "Mismatch in the total number of indices"

        log_bo_metrics(data_stats, dm.train_y, epoch=config["n_iters"])
        logger.setLevel(logging.INFO)
        wandb.finish()


def get_train_dimension(train_x):
    return train_x.shape[-1]


def main():
    # Initialize the parser with a description
    parser = ArgumentParser(
        description="Training script",
        default_config_files=["configs/bochemian.yaml"],
    )
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, help="Random seeds to use")
    parser.add_argument("--benchmark", type=str, help="Run a specific benchmark")

    parser.add_argument("--n_iters", type=int, help="How many iterations to run")
   
    parser.add_argument("--group", type=str, help="Wandb group runs")
    

    parser.add_subclass_arguments(BaseDataModule, "data", instantiate=False)
    parser.add_subclass_arguments(SurrogateModel, "surrogate_model", instantiate=False)

    parser.add_subclass_arguments(
        AcquisitionFunction,
        "acquisition",
        instantiate=False,
        skip=["model", "best_f"],
    )
    parser.add_subclass_arguments(BotorchOptimizer, "bo", instantiate=False)

    # parse arguments
    args = parser.parse_args()
    seed_everything(args["seed"], workers=True)
    train(args.as_dict())


if __name__ == "__main__":
    main()
