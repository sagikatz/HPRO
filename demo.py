import torch
import trimesh
import math
import numpy as np
from PointCloudComparer import PointCloudComparer
from GroundTruthGenerator import GroundTruthGenerator
from HPR import HPR
from HPRO import HPRO

if __name__ == '__main__':
    seed = 17
    np.random.seed(seed)

    use_linear_kernel = False

    viewpoint = [0.0, 0.0, -4.0]
    num_points = 10000
    gamma_hpr = -math.exp(-9.0)
    gamma_hpro = -math.exp(-10.0)

    # For large point clouds, the computation might not fit in memory
    # Use fits_in_memory = False to run sequentially with low memory requirements
    fits_in_memory = True

    # K from the paper
    k = 10

    # alpha from the paper
    alphas = []

    # delta from the paper
    delta = 0.0

    print("Loading mesh...")
    mesh = trimesh.load_mesh(R"lamp_0001.off")
    mesh.vertices -= np.mean(mesh.vertices, axis=0)
    mesh.vertices /= np.linalg.norm(mesh.vertices, axis=1).max()

    print("Sampling...")
    p, _ = trimesh.sample.sample_surface(mesh, count=num_points, face_weight=None, sample_color=False)

    print("Computing ground truth...")
    visible_pts_inds_gt = GroundTruthGenerator.compute(mesh, viewpoint, p)

    p = p.transpose(1, 0)
    p = np.expand_dims(p, axis=0)

    viewpoint = torch.tensor([viewpoint], dtype=torch.float64)
    pts = torch.tensor(p, dtype=torch.float64)

    print("HPRO...")
    hpro = HPRO(fits_in_memory=fits_in_memory).to('cuda')
    with torch.no_grad():
        pv_hpro, hpro_ind, _ = hpro(pts, viewpoint, gamma=gamma_hpro, alphas=alphas,
                                    k=k, delta=delta, use_linear_kernel=use_linear_kernel)

    hpro_ind = np.int32(hpro_ind.detach().cpu().numpy())

    print("HPR...")
    pv_hpr, pv_hpr_ind = HPR(pts, viewpoint, gamma_hpr, use_linear_kernel)

    # Evaluate the results
    comparer = PointCloudComparer()
    num_diff, only_in_pv_HPRO, only_in_pv_hpr = comparer.get_number_of_differences(hpro_ind, pv_hpr_ind)

    print('number of vertices in the model=' + str(mesh.vertices.shape[0]))
    print('number of points samples=' + str(p.shape[2]))
    print('number of differences=' + str(num_diff))
    print('only in HPRO=' + str(len(only_in_pv_HPRO)))
    print('only in HPR=' + str(len(only_in_pv_hpr)))

    num_errors_hpr, _, _ = comparer.get_number_of_differences_ind(visible_pts_inds_gt, pv_hpr_ind)
    print('HPR error=' + str(num_errors_hpr))
    num_errors_HPRO, _, _ = comparer.get_number_of_differences_ind(visible_pts_inds_gt, hpro_ind)
    print('HPRO error=' + str(num_errors_HPRO))

    # Plot the results of a comparison
    comparer.plot_results(viewpoint, visible_pts_inds_gt, pv_hpro, pv_hpr, only_in_pv_HPRO, only_in_pv_hpr, p)
