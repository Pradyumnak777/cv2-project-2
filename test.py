import numpy as np
from utils import get_3D_plane, visualize_3d_points

#load the points from the colmap text file
all_data = np.loadtxt("sparse_text/points3D.txt", usecols=(1, 2, 3, 4, 5, 6), comments="#")
pts = all_data[:, :3]
rgb = all_data[:, 3:]
#run ransac for a few thousand iterations
#threshold needs to be decided...will do trial n error..
threshold = np.ptp(pts, axis=0).max() * 0.01
# threshold = 0.05
table_inliers = get_3D_plane(pts, 2000, threshold)

#save the inlier indices so we can color them differently later
np.save("inlier_indices.npy", table_inliers)
print(f"found {len(table_inliers)} inliers for the dominant plane")

visualize_3d_points(pts, colors=rgb, inlier_indices=table_inliers)