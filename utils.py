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


def visualize_3d_points(pts, inlier_indices=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('persp')

    if inlier_indices is None or len(inlier_indices) == 0:
        outlier_pts = pts
    else:
        outlier_mask = np.ones(len(pts), dtype=bool)
        outlier_mask[inlier_indices] = False
        outlier_pts = pts[outlier_mask]

    ax.scatter(
        outlier_pts[:, 0],
        outlier_pts[:, 1],
        outlier_pts[:, 2],
        c='lightgray',
        s=1,
        alpha=0.45,
        label='outliers'
    )

    #if we found the plane, plot those points in a different color
    if inlier_indices is not None:
        inlier_pts = pts[inlier_indices]
        ax.scatter(
            inlier_pts[:, 0],
            inlier_pts[:, 1],
            inlier_pts[:, 2],
            c='red',
            s=3,
            alpha=0.9,
            label='dominant plane'
        )

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    centers = (mins + maxs) / 2.0
    max_span = np.max(maxs - mins)
    half = max_span / 2.0

    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)
    ax.set_box_aspect([1, 1, 1])

    #better default angle..
    ax.view_init(elev=18, azim=-55)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    fig.tight_layout()
    
    plt.show()    