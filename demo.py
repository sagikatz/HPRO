# This script computes and compares visibility results using the HPR and HPRO algorithms.
# Key Steps:
# 1. Load and normalize the mesh from an OFF file.
# 2. Sample points uniformly from the mesh surface.
# 3. Compute ground truth visibility indices using the provided viewpoint.
# 4. Apply the HPRO and HPR algorithms to compute visibility results.
# 5. Compare the results from HPRO and HPR against each other and the ground truth.
# 6. Output statistics and visualize the differences using a custom comparer class.
#
# Dependencies:
# - trimesh: For loading and processing 3D mesh models.
# - torch: For efficient numerical computations, particularly on GPUs.
# - numpy: For array manipulation and basic numerical operations.
#
# Ensure the required mesh file (lamp_0001.off) and auxiliary modules (e.g., HPRO, HPR) are available.

import torch
import trimesh
import math
import numpy as np
from PointCloudComparer import PointCloudComparer
from GroundTruthGenerator import GroundTruthGenerator
from HPR import HPR
from HPRO import HPRO

# Main script execution
if __name__ == '__main__':
    # Set random seed for reproducibility
    seed = 17
    np.random.seed(seed)

    # Configurations
    use_linear_kernel = False  # Whether to use a linear kernel for calculations
    viewpoint = [0.0, 0.0, -4.0]  # Viewpoint for visibility computations
    num_points = 10000  # Number of points to sample on the mesh surface
    gamma_hpr = -math.exp(-9.0)  # Gamma parameter for HPR algorithm
    gamma_hpro = -math.exp(-10.0)  # Gamma parameter for HPRO algorithm
    fits_in_memory = True  # Flag to determine whether computation fits in memory
    k = 10  # "K" parameter from the referenced paper
    alphas = []  # List of alpha values, configurable parameter from the paper
    delta = 0.0  # "Delta" parameter from the paper

    # Load mesh
    print("Loading mesh...")
    mesh = trimesh.load_mesh(R"lamp_0001.off")

    # Normalize mesh: Center and scale vertices
    mesh.vertices -= np.mean(mesh.vertices, axis=0)
    mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()

    # Sample points on the mesh surface
    print("Sampling...")
    p, _ = trimesh.sample.sample_surface(
        mesh, count=num_points, face_weight=None, sample_color=False
    )

    # Compute ground truth visibility indices
    print("Computing ground truth...")
    visible_pts_inds_gt = GroundTruthGenerator.compute(mesh, viewpoint, p)

    # Prepare point cloud data
    p = p.transpose(1, 0)  # Transpose for compatibility with algorithms
    p = np.expand_dims(p, axis=0)  # Expand dimensions for batch processing

    viewpoint = torch.tensor([viewpoint], dtype=torch.float64)  # Convert viewpoint to tensor
    pts = torch.tensor(p, dtype=torch.float64)  # Convert points to tensor

    # HPRO visibility computation
    print("HPRO...")
    hpro = HPRO(fits_in_memory=fits_in_memory).to('cuda')
    with torch.no_grad():
        pv_hpro, hpro_ind, _ = hpro(
            pts, viewpoint, gamma=gamma_hpro, alphas=alphas,
            k=k, delta=delta, use_linear_kernel=use_linear_kernel
        )
    hpro_ind = np.int32(hpro_ind.detach().cpu().numpy())  # Convert indices to NumPy array

    # HPR visibility computation
    print("HPR...")
    pv_hpr, pv_hpr_ind = HPR(pts, viewpoint, gamma_hpr, use_linear_kernel)

    # Evaluate the results by comparing HPRO and HPR outputs
    comparer = PointCloudComparer()
    num_diff, only_in_pv_HPRO, only_in_pv_hpr = comparer.get_number_of_differences(hpro_ind, pv_hpr_ind)

    # Print comparison statistics
    print(f'Number of vertices in the model: {mesh.vertices.shape[0]}')
    print(f'Number of points sampled: {p.shape[2]}')
    print(f'Number of differences: {num_diff}')
    print(f'Only in HPRO: {len(only_in_pv_HPRO)}')
    print(f'Only in HPR: {len(only_in_pv_hpr)}')

    # Evaluate HPR error against ground truth
    num_errors_hpr, _, _ = comparer.get_number_of_differences_ind(visible_pts_inds_gt, pv_hpr_ind)
    print(f'HPR error: {num_errors_hpr}')

    # Evaluate HPRO error against ground truth
    num_errors_hpro, _, _ = comparer.get_number_of_differences_ind(visible_pts_inds_gt, hpro_ind)
    print(f'HPRO error: {num_errors_hpro}')

    # Plot the comparison results
    comparer.plot_results(
        viewpoint, visible_pts_inds_gt, pv_hpro, pv_hpr,
        only_in_pv_HPRO, only_in_pv_hpr, p
    )



