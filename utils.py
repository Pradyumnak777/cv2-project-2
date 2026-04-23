import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
after getting the points3D.txt, write the RANSAC routine..
'''

def get_3D_plane(pts, iters, thresh):
    best_inliers = []
    
    for i in range(iters):
        #randomly sample 3 points to define a plane
        idx = random.sample(range(pts.shape[0]), 3)
        p1, p2, p3 = pts[idx]
        
        #find the normal vector using cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        a, b, c = normal
        
        #solve for d in ax + by + cz + d = 0
        d = -np.dot(normal, p1)
        
        #calculate the distance from every point to this plane
        #using the formula |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)...
        norm_mag = np.sqrt(a**2 + b**2 + c**2)
        if norm_mag < 1e-6: continue #skip if points are colinear
            
        dists = np.abs(a*pts[:,0] + b*pts[:,1] + c*pts[:,2] + d) / norm_mag
        
        #check which points are close enough to be inliers
        current_inliers = np.where(dists < thresh)[0]
        
        #keep the biggest group found so far
        if len(current_inliers) > len(best_inliers):
            best_inliers = current_inliers
            
    return best_inliers


# def visualize_3d_points(pts, inlier_indices=None):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='lightgray', s=1, label='outliers')

#     #if we found the plane, plot those points in a different color
#     if inlier_indices is not None:
#         inlier_pts = pts[inlier_indices]
#         ax.scatter(inlier_pts[:, 0], inlier_pts[:, 1], inlier_pts[:, 2], c='red', s=2, label='dominant plane')

#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.legend()
    
#     plt.show()    