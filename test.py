import numpy as np
from utils import get_3D_plane, visualize_3d_points

#load the points from the colmap text file
points = np.loadtxt("sparse_text/points3D.txt", usecols=(1, 2, 3))

#run ransac for a few thousand iterations
#threshold needs to be decided...will do trial n error..
table_inliers = get_3D_plane(points, 2000, 0.05)

#save the inlier indices so we can color them differently later
np.save("inlier_indices.npy", table_inliers)
print(f"found {len(table_inliers)} inliers for the dominant plane")

visualize_3d_points(points)