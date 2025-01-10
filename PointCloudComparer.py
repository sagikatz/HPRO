import numpy as np
import matplotlib.pyplot as plt



class PointCloudComparer:
    def __init__(self):
        pass

    def get_number_of_differences(self, indices1, indices2):
        """
        Compares two sets of point indices and returns the number of differences
        and the indices only found in one set.
        """
        # Convert indices to sets
        set1 = set(indices1)
        set2 = set(indices2)

        # Find differences between the two sets
        only_in_set1 = set1.difference(set2)
        only_in_set2 = set2.difference(set1)

        num_diff = len(only_in_set1) + len(only_in_set2)

        return num_diff, list(only_in_set1), list(only_in_set2)

    def get_number_of_differences_ind(self, ind1, ind2):
        """
        Compares two sets of indices and returns the number of differences
        and the indices only found in one set.
        """
        # Sort and convert to sets
        set1 = set(np.sort(ind1))
        set2 = set(np.sort(ind2))

        # Find differences
        only_in_set1 = set1.difference(set2)
        only_in_set2 = set2.difference(set1)

        num_diff = len(only_in_set1) + len(only_in_set2)

        return num_diff, list(only_in_set1), list(only_in_set2)

    def plot_results(self, viewpoint, visible_pts_inds_gt, pv_hpro, pv_hpr, only_in_pv_hpro, only_in_pv_hpr, p):
        """
        Plots the results of the comparison between ground truth, HPRO, and HPR.
        """
        cols, rows = 2, 3
        fig = plt.figure(figsize=(cols, rows))

        ax0 = fig.add_subplot(cols, rows, 1, projection="3d")
        ax0.scatter(p[0, 0, :], p[0, 1, :], p[0, 2, :], c='b', s=1)
        plt.xlim(-1.0, 1.0)
        plt.ylim(-1.0, 1.0)
        ax0.set_zlim(-1.0, 1.0)
        plt.title('Input Points')

        ax1 = fig.add_subplot(cols, rows, 2, projection="3d")
        ax1.scatter(p[0, 0, visible_pts_inds_gt], p[0, 1, visible_pts_inds_gt], p[0, 2, visible_pts_inds_gt], c='b',
                    s=1)
        ax1.scatter(viewpoint[0, 0].numpy(), viewpoint[0, 1].numpy(), viewpoint[0, 2].numpy(), c='r', s=10)
        plt.title('Ground Truth (red point is the viewpoint)')

        ax2 = fig.add_subplot(cols, rows, 3, projection="3d")
        ax2.scatter(pv_hpro[0, 0, :].cpu().numpy(), pv_hpro[0, 1, :].cpu().numpy(), pv_hpro[0, 2, :].cpu().numpy(),
                    c='b', s=1)
        ax2.scatter(viewpoint[0, 0].numpy(), viewpoint[0, 1].numpy(), viewpoint[0, 2].numpy(), c='r', s=1)
        plt.title('HPRO Points (red point is the viewpoint)')

        ax3 = fig.add_subplot(cols, rows, 4, projection="3d")
        ax3.scatter(pv_hpr[0, 0, :].numpy(), pv_hpr[0, 1, :].numpy(), pv_hpr[0, 2, :].numpy(), c='b', s=1)
        ax3.scatter(viewpoint[0, 0].numpy(), viewpoint[0, 1].numpy(), viewpoint[0, 2].numpy(), c='r', s=10)
        plt.title('HPR Points (red point is the viewpoint)')

        ax4 = fig.add_subplot(cols, rows, 5, projection="3d")
        ax4.scatter(p[0, 0, :], p[0, 1, :], p[0, 2, :], c='b', s=0.1)
        if len(only_in_pv_hpro) > 0:
            ax4.scatter(p[0, 0, only_in_pv_hpro], p[0, 1, only_in_pv_hpro], p[0, 2, only_in_pv_hpro], c='r', s=1)
        plt.title('Only in HPRO (red)')

        ax5 = fig.add_subplot(cols, rows, 6, projection="3d")
        ax5.scatter(p[0, 0, :], p[0, 1, :], p[0, 2, :], c='b', s=0.1)
        if len(only_in_pv_hpr) > 0:
            ax5.scatter(p[0, 0, only_in_pv_hpr], p[0, 1, only_in_pv_hpr], p[0, 2, only_in_pv_hpr], c='r', s=1)
        plt.title('Only in HPR (red)')

        plt.show()
