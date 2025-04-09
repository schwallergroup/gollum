import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional



class Featurizer:
    """
    Main featurizer class that handles different featurization methods.
    Uses a factory pattern to create the appropriate featurization function.
    """
    def __init__(
        self,
        representation: str = "fingerprints",
        # common parameters with defaults
        bond_radius: int = 3,
        nBits: int = 2048,
        model_name: Optional[str] = None,
        pooling_method: Optional[str] = None,
        normalize_embeddings: bool = False,
    ):
        self.representation = representation
        
        self.params = {
            "bond_radius": bond_radius,
            "nBits": nBits,
            "model_name": model_name,
            "pooling_method": pooling_method,
            "normalize_embeddings": normalize_embeddings,
        }
        self._featurization_registry = self._build_registry()
        self._output_dim = None
    
    def _build_registry(self):
        """
        Build a registry of featurization functions.
        This centralizes the import logic and makes it easier to add new methods.
        """
        from gollum.featurization.molecular import fingerprints, fragments, mqn_features, chemberta_features
        from gollum.featurization.text import get_tokens, get_huggingface_embeddings, instructor_embeddings
        from gollum.featurization.reaction import rxnfp, drfp, one_hot
        from gollum.featurization.general import precalculated, all_continuous
        
        return {
            "fingerprints": fingerprints,
            "fragments": fragments,
            "mqn_features": mqn_features,
            "chemberta_features": chemberta_features,
            "ohe": one_hot,
            "rxnfp": rxnfp,
            "drfp": drfp,
            "get_huggingface_embeddings": get_huggingface_embeddings,
            "get_tokens": get_tokens,
            "instructor_embeddings": instructor_embeddings,
            "precalculated": precalculated,
            "all_continuous": all_continuous,
        }
    
    @property
    def output_dim(self) -> int:
        """
        Returns the output dimension of the featurizer.
        This is determined after the first featurization.
        
        Returns:
            int: Output dimension
        """
        if self._output_dim is None:
            raise ValueError("Output dimension not yet determined. Run featurize() first.")
        return self._output_dim


    def featurize(self, data) -> np.ndarray:
        """
        Transform input data into feature vectors based on the configured representation.
        
        Args:
            data: Input data to featurize (list, pandas Series, etc.)
            
        Returns:
            np.ndarray: Featurized data
        """
        data_list = data.tolist() if hasattr(data, 'tolist') else data
        
        if self.representation not in self._featurization_registry:
            raise ValueError(f"Unsupported representation: {self.representation}")
        
        featurize_func = self._featurization_registry[self.representation]
        
        import inspect
        sig = inspect.signature(featurize_func)
        valid_params = {k: v for k, v in self.params.items() if k in sig.parameters}
        
        features = featurize_func(data_list, **valid_params)
        
        if isinstance(features, np.ndarray):
            self._output_dim = features.shape[1] if len(features.shape) > 1 else 1
        
        return features
