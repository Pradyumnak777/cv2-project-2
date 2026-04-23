import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

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


def visualize_3d_points(pts, colors=None, inlier_indices=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    
    if colors is None:
        colors = np.ones_like(pts) * 0.8
    else:
        colors = colors.astype(np.float64) / 255.0
    
    geometries = [pcd]
    
    #color the inliers bright red and draw the plane boundary
    if inlier_indices is not None and len(inlier_indices) > 0:
        colors[inlier_indices] = [1.0, 0.0, 0.0]
        
        #isolate the table points
        inlier_cloud = pcd.select_by_index(inlier_indices)
        
        obb = inlier_cloud.get_oriented_bounding_box()
        obb.color = (1.0, 0.0, 0.0)
        geometries.append(obb)
        
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    #draw both the points and the transparent plane boundary
    o3d.visualization.draw_geometries(geometries)