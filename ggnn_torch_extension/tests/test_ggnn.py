import torch
from ggnn_extension import GGNNTorch

def test_build_and_query():
    D = 128
    N = 1024
    k = 10
    device = 'cuda:0'
    
    base_data = torch.randn(N, D, device=device)
    
    model = GGNNTorch(k_neighbors=k).to(device)
    model.build(base_data)
    
    # Query with the first 5 vectors
    indices, distances = model(base_data[:5])
    
    assert indices.shape == (5, k + 1)
    assert distances.shape == (5, k + 1)
    # The first neighbor should be the point itself
    assert torch.all(indices[:, 0] == torch.arange(5, device=device))