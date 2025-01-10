import numpy as np
import trimesh


class GroundTruthGenerator:
    @staticmethod
    def singleRayIntersection(point_ind, intersector, viewpoint, sampled_locations):
        point_to_check = sampled_locations[point_ind, :]
        intersection, _, _ = intersector.intersects_location(ray_origins=[viewpoint], ray_directions=[point_to_check - viewpoint])
        first_intersection = np.argmin(np.linalg.norm(intersection - viewpoint, axis=1)).item()
        dis = np.linalg.norm(intersection[first_intersection] - point_to_check, axis=0)

        return dis < 1e-6

    @staticmethod
    def compute(mesh, viewpoint, sampled_locations):
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        visible_vertices_ind = []

        num_points = sampled_locations.shape[0]
        for point_ind in range(num_points):
            if GroundTruthGenerator.singleRayIntersection(point_ind, intersector, viewpoint, sampled_locations):
                visible_vertices_ind.append(point_ind)

        return visible_vertices_ind
