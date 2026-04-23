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
table_inliers, plane_normal = get_3D_plane(pts, 2000, threshold)

#save the inlier indices so we can color them differently later
np.save("inlier_indices.npy", table_inliers)
print(f"found {len(table_inliers)} inliers for the dominant plane")

# visualize_3d_points(pts, colors=rgb, inlier_indices=table_inliers)

'''
Now, make this plane the Z=0 axis, and "center" of this plane the origin of the new xyz coordinate system
'''

inlier_pts = pts[table_inliers] #points found by ransac
new_origin = np.mean(inlier_pts, axis=0) #o
z_local = plane_normal
z_local = plane_normal / np.linalg.norm(plane_normal)
dummy = np.array([1, 0, 0])
if np.abs(np.dot(z_local, dummy)) > 0.9: #handle edge case
    dummy = np.array([0, 1, 0])
    
x_local = np.cross(dummy, z_local)
x_local /= np.linalg.norm(x_local)
y_local = np.cross(z_local, x_local) #just cross prod.

#rotation matrix
R_local_to_scene = np.column_stack((x_local, y_local, z_local)) #stacking hte basis vectors as columns

print("transformation done. origin is centered in the inliers.")


'''
now, the icosahedron part..
'''
vertices = np.loadtxt('icosahedron.txt', comments=['#', 'f'], usecols=(1, 2, 3)) #need rgb too..
faces = np.loadtxt('icosahedron.txt', comments=['#', 'v'], usecols=(1, 2, 3), dtype=int) - 1

bottom_indices = np.where(np.abs(vertices[:, 2]) < 1e-6)[0] #checking z coordinate
bottom_center = np.mean(vertices[bottom_indices, :2], axis=0)

vertices[:, 0] -= bottom_center[0]
vertices[:, 1] -= bottom_center[1]

scale = 2.5 #adjust this until it looks right
vertices *= scale

scene_vertices = (R_local_to_scene @ vertices.T).T + new_origin
print("icosahedron placed and transformed into scene/physical space.")

#visualizing the icosahedron
# visualize_3d_points(pts, colors=rgb, inlier_indices=table_inliers, mesh_vertices=scene_vertices, mesh_faces=faces)