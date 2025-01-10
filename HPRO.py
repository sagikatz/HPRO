import torch
import torch.nn as nn

class HPRO(nn.Module):
    """
    Implementation of the HPRO (Hidden Point Removal for Optimization) algorithm.
    This model computes visibility for point clouds, optionally optimizing for memory or computational efficiency.

    Attributes:
        fits_in_memory (bool): Flag indicating if the computation fits in memory.
        If true, the code will parallelize the visiblity computation.
        If false, a lower memory footprint will be used, but with a longer compute time.
        visiblity_score_thresh (float): Threshold for visibility score to classify points as visible.
        It is kept at 0.99 in our experiments.
        device (str): Device on which the computations are performed (e.g., 'cuda', 'cpu').
    """

    def __init__(self, fits_in_memory=True, visiblity_score_thresh=0.99, device='cuda'):
        super(HPRO, self).__init__()
        self.elu = nn.ELU(alpha=1.0)
        self.visiblity_score_thresh = visiblity_score_thresh
        self.fits_in_memory = fits_in_memory
        self.device = device

    def detect_max_in_direction(self, batch_size, centered_points, directions, gamma, n_pts, use_linear_kernel,
                                alphas=None, k=10, delta=0.0):
        """
        Detects the maximum visibility response in a given direction.

        Args:
            batch_size (int): Number of batches.
            centered_points (torch.Tensor): Points centered around a viewpoint.
            directions (torch.Tensor): Normalized directions for each point.
            gamma (float): Transformation parameter for visibility calculations.
            n_pts (int): Number of points in the input point cloud.
            use_linear_kernel (bool): Flag indicating if a linear kernel is used.
            alphas (list): List of alpha values for additional view locations (see the paper).
            k (int): Number of top responses to consider.
            delta (float): Noise offset for transformed points.

        Returns:
            torch.Tensor: Visibility scores for each point.
        """
        # Compute transformed points based on gamma and direction
        norm = centered_points.norm(dim=1, keepdim=True)
        if use_linear_kernel:
            transformed_points_norm = (gamma - norm)
        else:
            transformed_points_norm = torch.pow(norm, gamma)

        transformed_points = transformed_points_norm * directions
        if delta > 0.0:
            transformed_points_with_noise_norm = torch.pow(norm - delta, gamma) if not use_linear_kernel else (
                    gamma - (norm - delta))
        else:
            transformed_points_with_noise_norm = transformed_points_norm

        if self.fits_in_memory:
            # Parallel computation for memory-fit scenario
            r_i = (transformed_points.repeat(1, 1, n_pts) * directions.repeat_interleave(n_pts, dim=2)).sum(dim=1,
                                                                                                            keepdim=True)
            r_i = r_i.reshape(batch_size, n_pts, n_pts)

            mk_k = torch.topk(r_i, k, dim=2, largest=True, sorted=True)
            w = (transformed_points_with_noise_norm - mk_k[0][:, :, k - 1]) / (mk_k[0][:, :, 0] - mk_k[0][:, :, k - 1])
            w = self.elu(w)
        else:
            # Memory-efficient computation
            w = torch.zeros((batch_size, n_pts), device=centered_points.device)
            for i in range(n_pts):
                r_i = (transformed_points * directions[:, :, [i]]).sum(dim=1)
                mk_k = torch.topk(r_i, k, dim=1, largest=True, sorted=True)
                w[:, i] = (transformed_points_with_noise_norm[:, :, i] - mk_k[0][:, k - 1]) / (
                            mk_k[0][:, 0] - mk_k[0][:, k - 1])
            w = self.elu(w)

        # Iterating through additional alphas for further transformations
        for alpha in alphas:
            center = transformed_points.mean(dim=2, keepdim=True) * alpha
            directions2 = torch.nn.functional.normalize(transformed_points - center, dim=1)

            if self.fits_in_memory:
                # Parallel computation for second transformation
                r_i_2 = ((transformed_points - center).repeat(1, 1, n_pts) * directions2.repeat_interleave(n_pts,
                                                                                                           dim=2)).sum(
                    dim=1, keepdim=True)
                r_i_2 = r_i_2.reshape(batch_size, n_pts, n_pts)

                mk_k_2 = torch.topk(r_i_2, k, dim=2, largest=True, sorted=True)
                r_i_2 = (transformed_points_with_noise_norm * directions - center).norm(dim=1, keepdim=False)
                w_2 = (r_i_2 - mk_k_2[0][:, :, k - 1]) / (mk_k_2[0][:, :, 0] - mk_k_2[0][:, :, k - 1])
                w_2 = self.elu(w_2)
            else:
                # Memory-efficient computation for second transformation
                w_2 = torch.zeros((batch_size, n_pts), device=centered_points.device)
                for i in range(n_pts):
                    r_i_2 = (transformed_points * directions2[:, :, [i]]).sum(dim=1)
                    mk_k = torch.topk(r_i_2, k, dim=1, largest=True, sorted=True)
                    w_2[:, i] = (transformed_points_with_noise_norm[:, :, i] - mk_k[0][:, k - 1]) / (
                                mk_k[0][:, 0] - mk_k[0][:, k - 1])
                w_2 = self.elu(w_2)

            w = torch.max(w, w_2)

        return w

    def forward(self, pts, viewpoint, gamma, alphas=[], k=10, delta=0.0, use_linear_kernel=False):
        """
        Forward pass for HPRO computation.

        Args:
            pts (torch.Tensor): Input point cloud of shape (batch_size, point_dim, num_points).
            viewpoint (torch.Tensor): Viewpoint for visibility computation.
            gamma (float): Transformation parameter for visibility calculations.
            alphas (list): List of alpha values for additional transformations.
            k (int): Number of top responses to consider.
            delta (float): Noise offset for transformed points.
            use_linear_kernel (bool): Flag indicating if a linear kernel is used.

        Returns:
            torch.Tensor: Visible points.
            torch.Tensor: Indices of visible points.
            torch.Tensor: Visibility scores (for optimization).
        """
        pts = pts.to(self.device)
        viewpoint = viewpoint.to(self.device)

        n_pts = pts.size()[2]
        pt_dim = pts.shape[1]
        batch_size = pts.shape[0]

        if len(viewpoint.shape) < 3:
            viewpoint = viewpoint.unsqueeze(dim=2)

        # Center the points around the viewpoint
        centered_points = pts - viewpoint.repeat_interleave(n_pts, dim=2)
        directions = torch.nn.functional.normalize(centered_points, dim=1)

        w = self.detect_max_in_direction(batch_size, centered_points, directions, gamma, n_pts, use_linear_kernel,
                                         k=k, alphas=alphas, delta=delta)

        # Threshold the visibility score to get the detected visible points
        w1 = w > self.visiblity_score_thresh

        # Return the detected points and indices as well as the differential values to be used for optimization
        return pts[:, :, w1.squeeze()].to(self.device), torch.nonzero(w1.squeeze()).squeeze().to(self.device), w
