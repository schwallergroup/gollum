import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA


def get_original_index(row, x):
    orig_index = np.where((x.numpy() == row).all(axis=1))[0]
    return orig_index[0]


def torch_delete_rows(input_tensor, indices):
    if indices is None:
        return input_tensor
    mask = torch.ones(input_tensor.size(0), dtype=torch.bool)
    mask[indices] = False
    return input_tensor[mask]


class Initializer(ABC):
    @abstractmethod
    def fit(self, x, exclude=None):
        pass


class MaxMinInitializer(Initializer):
    def __init__(self, metric="jaccard", n_clusters=20, **kwargs):
        self.metric = metric
        self.n_clusters = n_clusters

    def fit(self, x, exclude=None):
        x_init = torch_delete_rows(x, exclude)
        distances = cdist(x_init, x_init, metric=self.metric)
        selected_indices = [random.randint(0, distances.shape[0] - 1)]
        for _ in range(self.n_clusters - 1):
            coverage = np.min(distances[:, selected_indices], axis=1)
            selected_index = np.argmax(coverage)
            selected_indices.append(selected_index)
        return selected_indices, {}


class RandomInitializer(Initializer):
    def __init__(self, n_clusters=20, seed=0, **kwargs):
        self.n_clusters = n_clusters
        self.seed = seed

    def fit(self, x, exclude=None):
        return (
            np.random.default_rng(seed=self.seed)
            .permutation(len(x))[: self.n_clusters]
            .tolist(),
            {},
        )


class KMeansInitializer(Initializer):
    def __init__(
        self, n_clusters=20, seed=None, init_method="random", use_pca=False, **kwargs
    ):
        self.n_clusters = n_clusters
        self.seed = seed
        self.init_method = init_method
        self.use_pca = use_pca

    def fit(self, x, exclude=None):
        x_init = torch_delete_rows(x, exclude)
        if self.use_pca:
            print("using pca")
            pca = PCA(n_components=self.use_pca, random_state=self.seed)
            x_init = pca.fit_transform(x_init)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            max_iter=5000,
            init=self.init_method,
        ).fit(x_init)

        centroids = kmeans.cluster_centers_

        distances_to_centroids = np.zeros((len(x_init), len(centroids)))
        for i, centroid in enumerate(centroids):
            distances_to_centroids[:, i] = np.linalg.norm(x_init - centroid, axis=1)

        center_indices = []
        for i in range(len(centroids)):
            min_distance_index = np.argmin(distances_to_centroids[:, i])
            while min_distance_index in center_indices:
                distances_to_centroids[min_distance_index, i] = np.inf
                min_distance_index = np.argmin(distances_to_centroids[:, i])
            center_indices.append(min_distance_index)

        labels = kmeans.labels_
        clusters = {}

        for label in range(len(center_indices)):
            center_index = center_indices[label]
            distances = np.linalg.norm(x_init - x_init[center_index, :], axis=1)
            cluster_indices = np.where(labels == label)[0]
            sorted_indices = sorted(cluster_indices, key=lambda i: distances[i])
            clusters[label] = sorted_indices

        return center_indices, clusters


class PredefinedInitializer(Initializer):
    def __init__(self, predefined_points=None, **kwargs):
        self.predefined_points = predefined_points

    def fit(self, x, exclude=None):
        # use predefined points as the selected reactions
        selected_reactions = self.predefined_points
        clusters = {}
        return selected_reactions, clusters


class KMedoidsInitializer(Initializer):
    def __init__(
        self,
        n_clusters=20,
        seed=None,
        metric="jaccard",
        init_method="random",
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.seed = seed
        self.metric = metric
        self.init_method = init_method

    def fit(self, x, exclude=None):
        x_init = torch_delete_rows(x, exclude)
        kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            init=self.init_method,
            random_state=self.seed,
            metric=self.metric,
            max_iter=5000,
        ).fit(x_init)

        labels = kmedoids.labels_
        cluster_centers_indices = kmedoids.medoid_indices_
        clusters = {}

        for label in range(len(cluster_centers_indices)):
            center_index = cluster_centers_indices[label]
            distances = np.linalg.norm(x_init - x_init[center_index, :], axis=1)
            cluster_indices = np.where(labels == label)[0]
            sorted_indices = sorted(cluster_indices, key=lambda i: distances[i])
            clusters[label] = sorted_indices

        return kmedoids.medoid_indices_.tolist(), clusters


class BOInitializer:
    def __init__(
        self,
        method: str = "kmedoids",
        metric: str = None,
        n_clusters: int = None,
        init: str = "random",
        use_pca: int = None,
        seed: int = None,
        predefined_points: List[int] = None,
    ):
        self.seed = seed
        self.n_clusters = n_clusters
        self.methods = {
            "predefined": PredefinedInitializer,
            "maxmin": MaxMinInitializer,
            "true_random": RandomInitializer,
            "kmeans": KMeansInitializer,
            "kmedoids": KMedoidsInitializer,
        }
        if method not in self.methods:
            raise ValueError(f"Unknown init_method: {method}")
        self.initializer = self.methods[method](
            metric=metric,
            n_clusters=n_clusters,
            init_method=init,
            use_pca=use_pca,
            seed=self.seed,
            predefined_points=predefined_points,
        )
        self.selected_reactions = None

    def fit(self, x, exclude: list = None):
        x_init = torch_delete_rows(x, exclude)
        selected_reactions, clusters = self.initializer.fit(x_init, exclude)
        self.selected_reactions = [
            get_original_index(x_init.numpy()[i], x) for i in selected_reactions
        ]
        self.clusters = {}
        for label, indices in clusters.items():
            self.clusters[label] = [
                get_original_index(x_init.numpy()[i], x) for i in indices
            ]
        return self.selected_reactions, self.clusters
