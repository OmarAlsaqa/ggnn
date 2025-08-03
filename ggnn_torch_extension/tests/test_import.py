import torch

# The name here must match the 'name' in your setup.py and the directory you created
import ggnn_extension

# Check that the module and its C++ backend are loaded
print("Successfully imported ggnn_extension!")
print("Backend functions:", dir(ggnn_extension._C))

# --- Now, let's run a simple functional test ---

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available. This extension requires a GPU.")
else:
    print("CUDA is available. Running a test...")
    try:
        # Create some dummy data on the GPU
        D = 32  # Dimensions
        N_base = 1024  # Number of points in the dataset
        N_query = 10  # Number of query points
        K = 5  # Number of neighbors to find

        base_tensor = torch.randn(N_base, D, dtype=torch.float32).cuda()
        query_tensor = torch.randn(N_query, D, dtype=torch.float32).cuda()

        # Test the brute-force search function
        print("\nTesting brute-force search...")
        indices, distances = ggnn_extension.GGNNTorch.brute_force_search(
            base_tensor=base_tensor,
            query_tensor=query_tensor,
            k=K,
            measure=ggnn_extension.DistanceMeasure.Euclidean,
        )

        print("Brute-force search completed!")
        print("Indices shape:", indices.shape)  # Should be [10, 5]
        print("Distances shape:", distances.shape)  # Should be [10, 5]
        print("Indices dtype:", indices.dtype)
        print("Distances dtype:", distances.dtype)
        print("Indices device:", indices.device)

        # You are now ready to test the graph build and query as well!

    except Exception as e:
        print("\nAn error occurred during the test run:")
        print(e)
