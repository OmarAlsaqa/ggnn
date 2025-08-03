import torch
import ggnn_extension


# --- Helper function for evaluation ---
# Since the C++ Evaluator was not ported, we can write a simple and
# efficient recall calculator directly in PyTorch.
def calculate_recall(ground_truth_indices, result_indices):
    """Calculates the recall@k for the query results."""
    k = ground_truth_indices.shape[1]
    # For each query, count how many of its ground truth neighbors
    # are present in the result set.
    matches = torch.any(
        result_indices.unsqueeze(2) == ground_truth_indices.unsqueeze(1), dim=2
    ).sum(dim=1)
    # The recall is the average number of matches divided by k.
    recall = matches.float().mean() / k
    return recall.item()


# --- Main Script ---

# 2. Your extension is designed for the GPU. Ensure data is on the correct device.
if not torch.cuda.is_available():
    raise RuntimeError("This library requires a CUDA-enabled GPU.")
device = "cuda"

# Create data directly on the GPU
base = torch.rand((10_000, 128), dtype=torch.float32, device=device)
query = torch.rand((10_000, 128), dtype=torch.float32, device=device)

# 3. Initialize your new GGNNTorch module.
#    Build parameters are now passed to the constructor.
model = ggnn_extension.GGNNTorch(k_build=24, tau_build=0.5, refinement_iterations=2).to(
    device
)  # Move the module to the GPU

# 4. Choose a distance measure from the package.
measure = ggnn_extension.DistanceMeasure.Euclidean

# 5. Build the graph. This is now a method of your module.
#    It takes the base tensor and stores it and the graph internally.
model.build(base, measure=measure)


# 6. Run the query using the model's forward pass.
#    The model is callable, just like any other torch.nn.Module.
k_query: int = 10
tau_query: float = 0.64
max_iterations: int = 400

indices, dists = model(
    query_tensor=query,
    k_query=k_query,
    tau_query=tau_query,
    max_iterations=max_iterations,
)


# 7. Run brute-force query using the static method.
#    This doesn't require a built graph, just the base and query tensors.
gt_indices, gt_dists = ggnn_extension.GGNNTorch.brute_force_search(
    base_tensor=base, query_tensor=query, k=k_query, measure=measure
)

# 8. Evaluate the results using our pure Python helper function.
recall = calculate_recall(gt_indices, indices)
print(f"Recall@{k_query}: {recall:.4f}")


# 9. Print the results. Note that the library now correctly returns
#    actual distances, not squared distances.
print("\n--- Sample Results ---")
print("Indices:\n", indices[:5], "\n")
print("Distances:\n", dists[:5], "\n")
