import pytorch_lightning as pl
import pandas as pd
import numpy as np
import torch
from typing import Optional, Union, List

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from gollum.data.dataset import SingleSampleDataset
from gollum.data.utils import torch_delete_rows
from gollum.initialization.initializers import BOInitializer
from gollum.data.utils import find_duplicates, find_nan_rows
from gollum.featurization.base import Featurizer
from abc import ABC
import os
os.environ["OMP_NUM_THREADS"] = "28"


class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        data_path: str,
        input_column: Union[str, List[str]] = "input",
        target_column: str = "target",
        maximize: bool = True,
        init_sample_size: int = 10,
        featurizer: Featurizer = Featurizer(),
        initializer: BOInitializer = None,
        exclude_top: bool = False,
        normalize_input: str = "standard_scaling",
    ) -> None:
        self.data_path = data_path
        self.target_column = target_column
        self.input_column = input_column
        self.init_sample_size = init_sample_size
        self.featurizer = featurizer
        self.initializer = (
            initializer
            if initializer is not None
            else BOInitializer(method="true_random", n_clusters=init_sample_size)
        )
        self.exclude_top = exclude_top
        self.normalize_input = normalize_input
        self.maximize = maximize
        self.setup()

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        if not self.maximize:
            self.data[self.target_column] = -self.data[self.target_column]

    def featurize_data(self):
        x = self.featurizer.featurize(self.data[self.input_column])
        y = self.data[self.target_column].values

        self.x = torch.from_numpy(x).to(torch.float64)
        self.y = torch.from_numpy(y).to(torch.float64).unsqueeze(-1)

    def preprocess_data(self):
        nan_rows = find_nan_rows(self.x)
        self.x = torch_delete_rows(self.x, nan_rows)
        self.y = torch_delete_rows(self.y, nan_rows)
        self.data = self.data.drop(self.data.index[nan_rows]).reset_index(drop=True)

        duplicates = find_duplicates(self.x)
        self.x = torch_delete_rows(self.x, duplicates)
        self.y = torch_delete_rows(self.y, duplicates)
        self.data = self.data.drop(self.data.index[duplicates]).reset_index(drop=True)

        self.original_indices = np.arange(len(self.x))

        print(f"Removed {len(nan_rows)} nan rows")
        print(f"Removed {len(duplicates)} duplicate rows")

    def split_data(self):
        if self.exclude_top:
            indices = torch.arange(len(self.y))
            median_value = self.y.median()
            condition = self.y > median_value
            self.exclude = indices[condition.squeeze()].tolist()
        else:
            self.exclude = None

        init_indexes, _ = self.initializer.fit(self.x, exclude=self.exclude)
        print(f"Selected reactions: {init_indexes}")

        train_df_indices = self.original_indices[init_indexes].tolist()
        self.train_indexes = (
            init_indexes  
        )

        all_index_set = set(self.original_indices.tolist())
        train_index_set = set(train_df_indices)
        heldout_df_indices = list(all_index_set - train_index_set)

       
        self.train_x = self.x[init_indexes]
        self.train_y = self.y[init_indexes]

        index_to_position = {
            idx.item(): pos for pos, idx in enumerate(self.original_indices)
        }
        heldout_positions = [index_to_position[idx] for idx in heldout_df_indices]

        self.heldout_x = self.x[heldout_positions]
        self.heldout_y = self.y[heldout_positions]
        self.heldout_indices = torch.tensor(heldout_df_indices)

        sorted_indices = torch.argsort(self.heldout_y.squeeze())

        self.heldout_x = self.heldout_x[sorted_indices]
        self.heldout_y = self.heldout_y[sorted_indices]
        self.heldout_indices = self.heldout_indices[sorted_indices]

        train_df = self.data.loc[train_df_indices].copy()
        heldout_df = self.data.loc[self.heldout_indices.tolist()].copy()
        self.data = pd.concat([train_df, heldout_df])
        

    def update_results(self, experiment_results, experiment_indices):
        if self.target_column not in self.data.columns:
            self.data[self.target_column] = 0.0

        
        self.y[experiment_indices] = experiment_results

        self.train_y = torch.cat([self.train_y, experiment_results], dim=0)
        self.train_x = torch.cat(
            [self.train_x, self.heldout_x[experiment_indices]], dim=0
        )

        self.heldout_x = torch_delete_rows(self.heldout_x, experiment_indices)
        self.heldout_y = torch_delete_rows(self.heldout_y, experiment_indices)

        self.train_indexes = torch.cat([self.train_indexes, experiment_indices], dim=0)

    def normalize_data(self):
        def standard_scaling(X):
            return (X - X.mean()) / X.std()

        def l2_max_scaling(X):
            return X / torch.norm(X, dim=1).max()

        def l2_normalize(X):
            return X / torch.norm(X, dim=1, keepdim=True)

        if self.normalize_input == "standard_scaling":
            self.x = standard_scaling(self.x)
        elif self.normalize_input == "l2_max_scaling":
            self.x = l2_max_scaling(self.x)
        elif self.normalize_input == "l2_normalize":
            self.x = l2_normalize(self.x)
        elif self.normalize_input == "original":
            pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.load_data()
        self.featurize_data()
        self.preprocess_data()
        self.normalize_data()
        self.split_data()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = SingleSampleDataset(self.train_x, self.train_y)
        return DataLoader(train_dataset, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_dataset = SingleSampleDataset(self.heldout_x, self.heldout_y)
        return DataLoader(valid_dataset, num_workers=4)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = [item.to(device) for item in batch]
        return batch
