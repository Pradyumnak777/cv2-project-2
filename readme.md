# CSE 586 Project 2

### Instructions 
- First run `brew install colmap` on mac
- On Windows/Linux, check - https://colmap.github.io/install.html. 
- on terminal, type `colmap_gui`
- Then clcik on **import model**, and open the folder under `\sparse`
- Now, build the environment using `requirements.txt`.
- **Note!!!:** Make sure your PC/Mac has `ffmpeg` installed. On Mac, the command is `brew install ffmpeg`. 
- Run `main.py` to see the final `ar_output.mp4`
- If you want to visualize the intermediate steps, uncomment `visualize_3d_points(..)` function call after the RANSAC part to
visualize the point cloud and dominant plane
- To visualize the icosahedron in 3D space, uncomment the second `visualize_3d_points(...)` function call. Looking at the comments, finding this should be straighforward.
