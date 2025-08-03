# FILE: run_ggnn_metis_refactored.py

import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from skimage.segmentation import find_boundaries
import time

# --- Library Imports for this pipeline ---
# 1. Import your new, refactored library
import ggnn_extension
from torch_geometric.utils import sort_edge_index
from pyg_lib.partition import metis as pyg_metis

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
patch_size = 16
k_neighbors = 9
n_clusters_metis = 4
# ────────────────────────────────────────────────────────────────────────────────

def create_enhanced_visualization(
    base_image, cluster_labels, Hf, Wf, patch_size, cmap="Set1", alpha=0.6
):
    # This visualization function remains exactly the same
    label_map_small = cluster_labels.reshape(Hf, Wf)
    full_label_map = np.kron(
        label_map_small, np.ones((patch_size, patch_size), dtype=label_map_small.dtype)
    )
    colormap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=cluster_labels.min(), vmax=cluster_labels.max())
    colored_overlay_rgb = colormap(norm(full_label_map))[:, :, :3]
    final_image = (1 - alpha) * base_image + alpha * colored_overlay_rgb
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        cluster_color = colormap(norm(label))[:3]
        mask = full_label_map == label
        boundaries = find_boundaries(mask, mode="inner")
        final_image[boundaries] = cluster_color
    return final_image


# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    raise RuntimeError("This script requires a CUDA-enabled GPU.")

# Load image and model (this part is identical)
img_raw = data.astronaut()
img_np = resize(img_raw, (224, 224), anti_aliasing=True).astype(np.float32)
img_torch = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
img_vis = img_torch.squeeze(0).permute(1, 2, 0).cpu().numpy()
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
layers = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1]
if patch_size >= 8:
    layers.append(resnet.layer2)
if patch_size >= 16:
    layers.append(resnet.layer3)
backbone = nn.Sequential(*layers).eval().to(device)

# Extract features once (this part is identical)
with torch.no_grad():
    feat_map = backbone(img_torch)
B, C, Hf, Wf = feat_map.shape
num_nodes = Hf * Wf
patch_feats_gpu = feat_map.permute(0, 2, 3, 1).reshape(num_nodes, C).contiguous()

print(f"Feature map shape: {feat_map.shape}")
print(f"Total nodes: {num_nodes}, Nodes per image: {Hf * Wf}")

# --- Benchmarking Loop ---
print("\n--- Pipeline: Refactored GGNN + METIS ---")
start_time = time.perf_counter()

# 2. Initialize your new GGNNTorch module with build parameters.
#    The `k_build` parameter is used here.
model = ggnn_extension.GGNNTorch(k_build=k_neighbors).to(device)

# 3. Define the distance measure.
measure = ggnn_extension.DistanceMeasure.Euclidean

# 4. Build the graph. It's now a simple method call on the module.
model.build(patch_feats_gpu, measure=measure)

# 5. Run the query using the model's forward pass.
#    We query for k+1 neighbors because the first neighbor is always the node itself.
neighbor_indices, _ = model(
    query_tensor=patch_feats_gpu, k_query=k_neighbors + 1, tau_query=1.0
)

# 6. Construct the edge index for PyG. This logic remains the same.
#    The result `neighbor_indices` is already a GPU tensor.
src_ggnn = torch.arange(num_nodes, device=device).repeat_interleave(k_neighbors)
# Exclude the first neighbor (self-loop) and flatten
dst_ggnn = neighbor_indices[:, 1:].reshape(-1)
edge_index_ggnn = torch.stack([src_ggnn, dst_ggnn])

# 7. METIS requires the graph on the CPU.
edge_index_cpu = edge_index_ggnn.cpu()
row, col = sort_edge_index(edge_index_cpu, num_nodes=num_nodes)
counts = torch.bincount(row, minlength=num_nodes)
rowptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
metis_clusters = pyg_metis(rowptr, col, n_clusters_metis, recursive=False).numpy()

end_time = time.perf_counter()

print(f"Run time: {end_time - start_time:.4f} seconds")

# --- Visualization (this part is identical) ---
final_image = create_enhanced_visualization(img_vis, metis_clusters, Hf, Wf, patch_size)
plt.figure(figsize=(8, 8))
plt.imshow(final_image)
plt.title(
    f"Pipeline: Refactored GGNN + METIS\n({metis_clusters.max()+1} Fixed Clusters)"
)
plt.axis("off")
plt.show()
