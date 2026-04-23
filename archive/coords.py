import numpy as np

# Define the path to the text file generated in your previous script
file_path = 'sparse_text/points3D.txt'

# Read the file directly into a numpy array
# comments='#' automatically ignores the COLMAP header lines
# usecols=(1, 2, 3) isolates just the X, Y, and Z columns
xyz_coordinates = np.loadtxt(file_path, comments='#', usecols=(1, 2, 3))

# Print the shape to verify extraction (should be N rows by 3 columns)
print(xyz_coordinates.shape)
