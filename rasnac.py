import numpy as np
import matplotlib.pyplot as plt

# Assuming 'xyz_coordinates' is the (11353, 3) array from Step 3
file_path = 'sparse_text/points3D.txt'

# Read the file directly into a numpy array
# comments='#' automatically ignores the COLMAP header lines
# usecols=(1, 2, 3) isolates just the X, Y, and Z columns
xyz_coordinates = np.loadtxt(file_path, comments='#', usecols=(1, 2, 3))
pts = xyz_coordinates
num_points = pts.shape[0]

# RANSAC Hyperparameters
num_iters = 2000  # Number of random planes to test simultaneously
threshold = np.ptp(pts, axis=0).max() * 0.01 # Distance threshold (adjust based on the scale of your COLMAP model)

# 1. Randomly sample 3 points for all 2000 iterations simultaneously
random_indices = np.random.randint(0, num_points, size=(num_iters, 3))
p1 = pts[random_indices[:, 0]]
p2 = pts[random_indices[:, 1]]
p3 = pts[random_indices[:, 2]]

# 2. Fit the plane equation: ax + by + cz + d = 0
v1 = p2 - p1
v2 = p3 - p1

# Compute the normal vector via cross product
normals = np.cross(v1, v2)

# Normalize the normal vectors
norms = np.linalg.norm(normals, axis=1, keepdims=True)
# Add a tiny epsilon to prevent division by zero in case of perfectly collinear random points
normals = normals / (norms + 1e-10)

# Calculate 'd' (distance to origin)
d = -np.sum(normals * p1, axis=1, keepdims=True)

# 3. Calculate distances of ALL 11,353 points to ALL 2,000 planes simultaneously
# pts.T shape: (3, 11353) | normals shape: (2000, 3)
# The dot product yields a (2000, 11353) matrix of distances
distances = np.abs(np.dot(normals, pts.T) + d)

# 4. Determine inliers based on the threshold
inlier_masks = (distances < threshold) & (norms > 1e-6)

# 5. Keep track of the largest coplanar set
inlier_counts = np.sum(inlier_masks, axis=1)
best_plane_idx = np.argmax(inlier_counts)

# Extract the final inlier and outlier points for the best plane
best_mask = inlier_masks[best_plane_idx]
inliers = pts[best_mask]
outliers = pts[~best_mask]

# The best plane parameters (for later use in coordinate transformations)
best_normal = normals[best_plane_idx]
best_d = d[best_plane_idx]

# 6. Step 5: Display the 3D point cloud
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot outliers as tiny gray dots
ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], 
           c='gray', s=1, alpha=0.5, label='Scene Structure (Outliers)')

# Plot inliers (the dominant plane) as slightly larger red dots
ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], 
           c='red', s=5, label='Dominant Plane (Inliers)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title(f'RANSAC Dominant Plane Detection (Inliers: {inliers.shape[0]})')

# Display interactive plot 
# plt.show()

print(f"Total points: {num_points}")
print(f"Dominant plane inliers: {inliers.shape[0]}")

# --- STEP 6: Local Coordinate System Transformation ---

# 1. Choose the local origin: The mean (centroid) of the inlier points
centroid = np.mean(inliers, axis=0)

# 2. Define the local Z-axis as the normal of the dominant plane
# (Assuming 'best_normal' is carried over from your RANSAC script)
z_axis = best_normal / np.linalg.norm(best_normal)

# 3. Create a stable temporary vector to calculate the orthogonal X and Y axes
# We use np.argmin to find the component with the smallest absolute value 
# to guarantee non-collinearity without using any if-else conditional logic.
min_idx = np.argmin(np.abs(z_axis))
tmp_vector = np.zeros(3)
tmp_vector[min_idx] = 1.0

# 4. Compute orthogonal X and Y axes using cross products
x_axis = np.cross(tmp_vector, z_axis)
x_axis = x_axis / np.linalg.norm(x_axis)

y_axis = np.cross(z_axis, x_axis)
# y_axis is already a unit vector because it is the cross product of two orthogonal unit vectors

# 5. Form the Rotation Matrix (R)
# Stacking the axes as columns defines the rotation from local to global coordinates
R_local_to_global = np.column_stack((x_axis, y_axis, z_axis))

# The Translation vector (T) is simply the centroid
T_local_to_global = centroid

print("Rotation Matrix (Local to Global):")
print(R_local_to_global)
print("\nTranslation Vector (Local Origin):")
print(T_local_to_global)

# --- STEP 7: Virtual Object Transformation & Visualization ---

# 1. Load the Icosahedron Data using vectorized filtering
# Treat '#' and 'f' as comments so face lines are ignored entirely.
# usecols=(1, 2, 3) skips the 'v' string in column 0 and only reads the X, Y, Z floats.
vertices = np.loadtxt('icosahedron.txt', comments=['#', 'f'], usecols=(1, 2, 3))

# Treat '#' and 'v' as comments so vertex lines are ignored entirely.
# Subtract 1 from the entire array to convert the file's 1-based indexing into Python's 0-based indexing.
faces = np.loadtxt('icosahedron.txt', comments=['#', 'v'], usecols=(1, 2, 3), dtype=int) - 1

# 2. Translate the Object
# We need to find the bottom face (where z is roughly 0) to center it.
z_coords = vertices[:, 2]
bottom_vertices_mask = np.abs(z_coords) < 1e-4

# Extract the bottom vertices and calculate their geometric center
bottom_vertices = vertices[bottom_vertices_mask]
bottom_center = np.mean(bottom_vertices, axis=0)

# Subtract this center from all vertices to shift the object
vertices_centered = vertices - bottom_center

# 3. Scale the Object
scale_factor = 2.0
vertices_scaled = vertices_centered * scale_factor

# 4. Transform to Global Scene Coordinates
# P_scene = (R * P_local) + T
vertices_scene = np.dot(vertices_scaled, R_local_to_global.T) + T_local_to_global

# 5. Visualizing the Wireframe Mesh
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the dominant plane inliers to serve as a reference tabletop
ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], 
           c='gray', s=1, alpha=0.3, label='Dominant Plane')

# Plot the icosahedron using the 0-indexed faces array
ax.plot_trisurf(vertices_scene[:, 0], vertices_scene[:, 1], vertices_scene[:, 2], 
                triangles=faces, color='blue', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Virtual Icosahedron Placed on Dominant Plane')
plt.legend()

plt.show()

import numpy as np
from scipy.spatial.transform import Rotation

# --- STEP 8: Read Camera Parameters ---

# 1. Read Internal Parameters from cameras.txt
# Format: CAMERA_ID, MODEL_STRING, WIDTH, HEIGHT, PARAMS[F, CX, CY, K]
# We use usecols to skip the string columns and only grab the numeric data
cam_data = np.loadtxt('sparse_text/cameras.txt', comments='#', usecols=(2, 3, 4, 5, 6, 7))

# Ensure it's a 1D array (in case COLMAP output slightly differently)
cam_data = np.squeeze(cam_data)

width = cam_data[0]
height = cam_data[1]
focal_length = cam_data[2]
cx = cam_data[3]
cy = cam_data[4]
radial_distortion = cam_data[5]

print(f"Camera Internal Params -> Focal: {focal_length}, Center: ({cx}, {cy})")

# 2. Read External Parameters from images.txt
# images.txt alternates between pose data lines and 2D point data lines.
# We read the raw text and use array slicing to isolate just the pose lines, completely avoiding loops.
with open('sparse_text/images.txt', 'r') as file:
    lines = file.read().strip().split('\n')

# The first 4 lines are comments (indices 0, 1, 2, 3)
# Pose lines are at even indices starting at 4 (4, 6, 8...)
pose_lines = lines[4::2]

# Parse the pose lines into a numpy array
# Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME_STRING
# We usecols to grab only the quaternion and translation floats, ignoring strings
external_data = np.loadtxt(pose_lines, usecols=(0, 1, 2, 3, 4, 5, 6, 7))

# Extract the relevant columns
image_ids = external_data[:, 0]
quaternions_wxyz = external_data[:, 1:5]
translations = external_data[:, 5:8]

# 3. Convert Quaternions to Rotation Matrices
# SciPy's Rotation expects quaternions in scalar-last format (x, y, z, w)
# COLMAP outputs them in scalar-first format (w, x, y, z)
# We use numpy column stacking to rearrange them vectorially
quaternions_xyzw = np.column_stack((quaternions_wxyz[:, 1:4], quaternions_wxyz[:, 0]))

# Convert all quaternions to 3x3 rotation matrices simultaneously
rotations = Rotation.from_quat(quaternions_xyzw).as_matrix()

print(f"Successfully loaded {len(image_ids)} camera poses.")
print(f"Shape of Rotations array: {rotations.shape} (N, 3, 3)")
print(f"Shape of Translations array: {translations.shape} (N, 3)")

# --- STEP 9: 3D to 2D Camera Projection ---

# 1. Construct the Intrinsic Matrix (K) using parameters from Step 8
K = np.array([
    [focal_length, 0, cx],
    [0, focal_length, cy],
    [0, 0, 1]
])

# 2. Vectorized 3D to 2D Projection for ALL vertices across ALL 53 cameras!
# vertices_scene shape: (12, 3) 
# rotations shape: (53, 3, 3) 
# translations shape: (53, 3) 

# Transform points to camera coordinate system: X_cam = R * Xpy_world + T
# np.einsum multiplies the 53 rotation matrices (nij) with the 12 vertices (vj)
# yielding an output shape of (53 cameras, 12 vertices, 3 xyz coords) -> 'nvi'
points_rotated = np.einsum('nij,vj->nvi', rotations, vertices_scene)

# Add the translation vectors (broadcasting the (53, 3) array across the 12 vertices)
points_cam = points_rotated + translations[:, np.newaxis, :]

# Project onto the image plane: X_proj = K * X_cam
# Multiply the single 3x3 K matrix ('ij') with the transformed points ('nvj')
points_proj = np.einsum('ij,nvj->nvi', K, points_cam)

# Convert from homogeneous to 2D pixel coordinates by dividing by Z
# Add a microscopic epsilon to prevent division by zero
z_coords = points_proj[:, :, 2] + 1e-10

u_cols = points_proj[:, :, 0] / z_coords
v_rows = points_proj[:, :, 1] / z_coords

# Stack them back together into a final array of shape (53, 12, 2)
# This array now contains the 2D (col, row) pixel locations for every vertex in every frame!
pixel_coordinates = np.stack((u_cols, v_rows), axis=-1)

print(f"Projected pixel coordinates shape: {pixel_coordinates.shape} (Cameras, Vertices, XY)")


import matplotlib.patches as patches
from PIL import Image

# --- STEP 10: Augmented Reality Rendering (Painter's Algorithm) ---

# 1. Calculate Face Depths in Camera Space
# points_cam (from Step 9) holds the 3D coordinates relative to the camera
# Extract just the Z-coordinates (depths) for all vertices across all 53 cameras
z_cam = points_cam[:, :, 2]

# Map the Z-depths to the 3 vertices defining each of the 20 faces
# This creates a shape of (53 cameras, 20 faces, 3 vertices)
face_depths = z_cam[:, faces]

# Find the minimum depth for each face (the point closest to the camera)
min_face_depths = np.min(face_depths, axis=2)

# Sort faces from farthest to closest. 
# np.argsort sorts ascending, so passing negative depths forces a descending sort
face_draw_order = np.argsort(-min_face_depths, axis=1)

# 2. Extract image filenames 
# We use map with a lambda function to extract the filename from the end of each pose_line
# strictly avoiding list comprehensions or loops.
image_names = list(map(lambda line: line.split()[-1], pose_lines))

# 3. Create the loop-free drawing function for a single frame
def render_frame(i):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis('off') # Hide axes

    # Load and display the original background image
    img_path = f'images/{image_names[i]}'
    img = Image.open(img_path)
    ax.imshow(img)

    # Get the correctly sorted face indices for this specific frame
    ordered_face_indices = face_draw_order[i]

    # Get the 2D pixel coordinates for the faces in this frame
    camera_face_pixels = pixel_coordinates[i, faces, :]
    
    # Reorder the pixels based on our Z-depth sorting
    sorted_face_pixels = camera_face_pixels[ordered_face_indices]

    # Nested functional drawing routine for the faces
    def draw_polygon(face_idx_tuple):
        order_idx, face_pts = face_idx_tuple
        
        # Calculate a simple lighting shade based on the draw order 
        # Farthest faces are dark blue, closest faces are lighter blue
        shade = 0.3 + (0.7 * (order_idx / 19.0))
        color = (0.0, 0.0, shade)

        # Create and add the filled polygon patch
        poly = patches.Polygon(face_pts, closed=True, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(poly)

    # Map the polygon drawing function across all 20 sorted faces
    list(map(draw_polygon, enumerate(sorted_face_pixels)))

    # Save the final AR frame to the current directory
    out_name = f'AR_rendered_{image_names[i]}'
    plt.savefig(out_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Rendered {out_name}")
    
    return out_name

# 4. Map the frame rendering function across all 53 cameras
rendered_files = list(map(render_frame, range(len(image_names))))

print(f"\nSuccess! Rendered {len(rendered_files)} total Augmented Reality frames.")