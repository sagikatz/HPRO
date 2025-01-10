import torch
from scipy.spatial import ConvexHull


def HPR(pts, viewpoint, gamma,use_linear_kernel):
    # pts - bxdxn
    # viewpoint - 1xd
    # gamma = scalar
    n_pts = pts.size()[2]
    pt_dim = pts.shape[1]
    batch_size = pts.shape[0]

    if len(viewpoint.shape) < 3:
        viewpoint = viewpoint.unsqueeze(dim=2)

    # center the points around viewpoint
    centered_points = pts - viewpoint.repeat_interleave(n_pts, dim=2)

    directions = torch.nn.functional.normalize(centered_points, dim=1)

    # transform the points
    if use_linear_kernel:
        trans_points = (gamma - centered_points.norm(dim=1, keepdim=True)) * directions
    else:
        trans_points = torch.pow(centered_points.norm(dim=1, keepdim=True), gamma) * directions

    hull = ConvexHull(trans_points.cpu().detach().squeeze().permute(1, 0).numpy())

    return pts[:, :, hull.vertices], hull.vertices


