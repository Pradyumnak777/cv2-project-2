import os
import shutil

'''
I have an mp4 video, perform some preproc and use colmap to create point cloud
'''

# define the paths
video = "test_vid.mp4"
img_folder = "images"
db = "database.db"
sparse_out = "sparse"
text_out = "sparse_text"

# extract frames every 10th frame because 30fps is too much data
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

#remove old. KEEP IMAGE FOLDER!!!!!
if os.path.exists(db):
    os.remove(db)
if os.path.exists(sparse_out):
    shutil.rmtree(sparse_out)
if os.path.exists(text_out):
    shutil.rmtree(text_out)
'''
#NOTE: uncomment below if running straight from video!!
'''
# os.system(f"ffmpeg -i {video} -vf \"select='not(mod(n,10))'\" -vsync vfr {img_folder}/frame_%04d.jpg")

# simple_radial and shared, as same camera used, with same focal length
os.system(f"colmap feature_extractor --database_path {db} --image_path {img_folder} --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL")

# match the features between frames (sequential is great for video)
os.system(f"colmap sequential_matcher --database_path {db}")

# start the actual 3d reconstruction
if not os.path.exists(sparse_out):
    os.makedirs(sparse_out)

'''
below code prevents from generating multiple sparse folders...reduction of feature matching thresh..
'''

mapper_cmd = (
    f"colmap mapper "
    f"--database_path {db} "
    f"--image_path {img_folder} "
    f"--output_path {sparse_out} "
    f"--Mapper.multiple_models 0 "
    f"--Mapper.min_num_matches 10 "
    f"--Mapper.init_min_num_inliers 80"
)
os.system(mapper_cmd)

if not os.path.exists(text_out):
    os.makedirs(text_out)

os.system(f"colmap model_converter --input_path {sparse_out}/0 --output_path {text_out} --output_type TXT")

print("done. check the sparse_text folder for cameras.txt, images.txt and points3d.txt")