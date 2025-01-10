import torch
from scipy.spatial import ConvexHull

def HPR(pts, viewpoint, gamma, use_linear_kernel):
    """
    Hidden Point Removal (HPR) Algorithm for determining the direct visibility of point sets.

    Reference: Katz, S., Tal, A., & Basri, R. (2007). Direct visibility of point sets. ACM Transactions on Graphics (TOG), 26(3), 24-es.

    Parameters:
    - pts (torch.Tensor): A tensor of shape (batch_size, dimensions, num_points) representing the point cloud.
    - viewpoint (torch.Tensor): A tensor of shape (1, dimensions) representing the viewpoint.
    - gamma (float): Kernel parameter to control the transformation.
    - use_linear_kernel (bool): Flag to determine whether to use a linear or power kernel.

    Returns:
    - visible_points (torch.Tensor): The subset of input points that are directly visible.
    - visible_indices (numpy.ndarray): Indices of the visible points in the original input.
    """
    # Validate input dimensions
    batch_size, pt_dim, n_pts = pts.size()
    if len(viewpoint.shape) < 3:
        viewpoint = viewpoint.unsqueeze(dim=2)  # Ensure viewpoint has shape (1, dimensions, 1)

    # Center the points around the viewpoint
    centered_points = pts - viewpoint.repeat_interleave(n_pts, dim=2)

    # Normalize directions
    directions = torch.nn.functional.normalize(centered_points, dim=1)

    # Transform the points using the chosen kernel
    if use_linear_kernel:
        trans_points = (gamma - centered_points.norm(dim=1, keepdim=True)) * directions
    else:
        trans_points = torch.pow(centered_points.norm(dim=1, keepdim=True), gamma) * directions

    # Compute the convex hull of the transformed points (only supports CPU tensors)
    trans_points_np = trans_points.cpu().detach().squeeze().permute(1, 0).numpy()
    hull = ConvexHull(trans_points_np)

    # Extract the visible points and their indices
    visible_points = pts[:, :, hull.vertices]
    visible_indices = hull.vertices

    return visible_points, visible_indices
