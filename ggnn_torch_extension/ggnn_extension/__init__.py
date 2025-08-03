import torch
import torch.nn as nn
from typing import Tuple

# This is the crucial import. It brings in the C++ functions we bound in ggnn.cpp.
# The name '_C' should match what you have in setup.py.
from . import _C as _backend

# 2. Expose the user-facing enum directly at the package level.
#    This makes `ggnn_extension.DistanceMeasure` available to users.
DistanceMeasure = _backend.DistanceMeasure

# This is a list of the public-facing names that will be available
# when a user does 'from ggnn_extension import *'
__all__ = [
    "GGNNTorch",
    "DistanceMeasure",
]


class GGNNTorch(nn.Module):
    """
    A PyTorch module for GGNN: Graph-based GPU Nearest Neighbor Search.

    This module encapsulates the GGNN graph as a persistent buffer. You first
    `build()` the graph from a base dataset, and then you can perform fast
    queries by calling the module instance like a function.

    Args:
        k_build (int): The number of neighbors per point to use when constructing the graph.
                       This is a key parameter that balances graph quality and memory usage.
        tau_build (float, optional): The slack factor for graph construction, affecting the
                                     trade-off between build time and graph quality. Defaults to 1.0.
        refinement_iterations (int, optional): Number of refinement passes to improve graph
                                               quality after the initial build. Defaults to 2.
    """

    def __init__(
        self, k_build: int, tau_build: float = 1.0, refinement_iterations: int = 2
    ):
        super().__init__()

        # Store build-time configuration parameters
        self.k_build = k_build
        self.tau_build = tau_build
        self.refinement_iterations = refinement_iterations

        # The graph and base tensors will be stored here after 'build()' is called.
        # We register them as buffers, so PyTorch will manage their memory,
        # including moving them between devices with .to(device).
        self.register_buffer("graph_tensor", None)
        self.register_buffer("base_tensor", None)
        # Store the distance measure used for building
        self.build_measure = None

    def build(
        self,
        base_tensor: torch.Tensor,
        measure: DistanceMeasure = DistanceMeasure.Euclidean,
    ):
        """
        Builds the GGNN search graph from the base tensor and stores it in the module.
        This method must be called once before you can perform queries.

        Args:
            base_tensor (torch.Tensor): A 2D CUDA tensor of shape [N, D] containing the dataset.
            measure (ggnn_torch_backend.DistanceMeasure, optional): The distance metric to use.
                                                                    Defaults to Euclidean.
        """
        if self.graph_tensor is not None:
            print(
                "Warning: Graph is already built. Re-building will overwrite the existing graph."
            )

        # The base tensor needs to be stored for distance calculations during the query phase.
        self.base_tensor = base_tensor
        self.build_measure = measure

        # Call the C++ backend function to perform the heavy lifting on the GPU.
        # The function returns a raw byte tensor containing the graph data.
        self.graph_tensor = _backend.build_graph(
            base_tensor=self.base_tensor,
            k_build=self.k_build,
            tau_build=self.tau_build,
            refinement_iterations=self.refinement_iterations,
            measure=self.build_measure,
        )
        print(f"GGNN graph built successfully for tensor of shape {base_tensor.shape}.")

    def forward(
        self,
        query_tensor: torch.Tensor,
        k_query: int,
        tau_query: float = 1.0,
        max_iterations: int = 400,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a fast approximate nearest neighbor search on the pre-built graph.

        Args:
            query_tensor (torch.Tensor): A 2D CUDA tensor of shape [Q, D] of query vectors.
            k_query (int): The number of nearest neighbors to find for each query.
            tau_query (float, optional): The slack factor for querying. Higher values can
                                         increase accuracy at the cost of speed. Defaults to 1.0.
            max_iterations (int, optional): The maximum number of search steps. Defaults to 400.
            measure (ggnn_torch_backend.DistanceMeasure, optional): The distance metric to use.
                                                                    Defaults to Euclidean.

        Returns:
            A tuple of (indices, distances) tensors, both of shape [Q, k_query].
        """
        if self.graph_tensor is None or self.base_tensor is None:
            raise RuntimeError(
                "The graph has not been built. Please call .build(base_tensor) before querying."
            )

        # Call the C++ backend function for the query.
        # We pass the query data along with the graph and base tensors stored in the module.
        indices, distances = _backend.query_graph(
            query_tensor=query_tensor,
            graph_tensor=self.graph_tensor,
            base_tensor=self.base_tensor,
            k_build=self.k_build,
            k_query=k_query,
            tau_query=tau_query,
            max_iterations=max_iterations,
            measure=self.build_measure,
        )
        return indices, distances

    @staticmethod
    def brute_force_search(
        base_tensor: torch.Tensor,
        query_tensor: torch.Tensor,
        k: int,
        measure: DistanceMeasure = DistanceMeasure.Euclidean,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a brute-force (exact) k-NN search on the GPU.
        This is useful for generating ground truth data to evaluate the accuracy of the
        approximate search. This method is static as it does not depend on a built graph.

        Args:
            base_tensor (torch.Tensor): The [N, D] dataset tensor.
            query_tensor (torch.Tensor): The [Q, D] query tensor.
            k (int): The number of neighbors to find.
            measure (ggnn_torch_backend.DistanceMeasure, optional): The distance metric. Defaults to Euclidean.

        Returns:
            A tuple of (indices, distances) tensors, both of shape [Q, k].
        """
        return _backend.bf_query(
            base_tensor=base_tensor, query_tensor=query_tensor, k=k, measure=measure
        )
