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