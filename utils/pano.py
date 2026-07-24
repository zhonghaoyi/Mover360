from external import py360convert
import numpy as np
from PIL import Image
import os
import argparse
from external import PanoAnnotator
import torch.nn.functional as F
from einops import rearrange
import math
import copy
from tqdm import tqdm
from . import createpano
# Assumed default FOV (needs to be verified based on Matterport3D data or passed as parameter)
DEFAULT_FOV_DEGREES = 90.0 # Matterport undistorted images are often around 90-100 degrees FOV

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def random_sample_spherical(n):
    xyz = np.random.normal(size=(n, 3))
    xyz = normalize(xyz)
    return xyz


def random_sample_camera(n):
    xyz = random_sample_spherical(n)
    phi = np.arcsin(xyz[:, 2].clip(-1, 1))
    theta = np.arctan2(xyz[:, 0], xyz[:, 1])
    return theta, phi


def horizon_sample_camera(n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    phi = np.zeros_like(theta)
    return theta, phi

def cubemap_sample_camera():
    """
    Generate viewpoint parameters (theta, phi) for six faces of a cube
    Returns:
        thetas: Yaw angle array (in radians)
        phis: Pitch angle array (in radians)
    """
    # Six faces of the cube [front, right, back, left, up, down]
    # theta (yaw): 0=front, π/2=right, π=back, -π/2=left
    # phi (pitch): 0=horizontal, π/2=up, -π/2=down
    thetas = np.array([0, np.pi/2, np.pi, -np.pi/2, 0, 0])
    phis = np.array([0, 0, 0, 0, np.pi/2, -np.pi/2])
    
    return thetas, phis

def eight_pers_sample_camera():
    """
    Generate camera parameters (theta, phi) for eight viewpoints
    8 evenly distributed viewpoints in horizontal direction, plus up and down viewpoints
    Returns:
        thetas: Yaw angle array (in radians)
        phis: Pitch angle array (in radians)
    """
    # horizontal direction 8 evenly distributed viewpoints, interval 45 degrees (π/4)
    horizontal_thetas = np.linspace(0, 2*np.pi, 9)[:-1]  # 0 to 2π uniformly distributed 8 points, remove the last one (duplicate with the first)
    horizontal_phis = np.zeros(8)  # pitch angle of horizontal viewpoints are all 0
    
    # up and down viewpoints
    vertical_thetas = np.array([0, 0])
    vertical_phis = np.array([np.pi/2, -np.pi/2])  # up (π/2) and down (-π/2)
    
    # merge all viewpoints
    thetas = np.concatenate([horizontal_thetas, vertical_thetas])
    phis = np.concatenate([horizontal_phis, vertical_phis])
    
    return thetas, phis

def icosahedron_sample_camera():
    # reference: https://en.wikipedia.org/wiki/Regular_icosahedron
    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)
    theta_step = 2.0 * np.pi / 5.0

    thetas = []
    phis = []
    for triangle_index in range(20):
        # 1) the up 5 triangles
        if 0 <= triangle_index <= 4:
            theta = - np.pi + theta_step / 2.0 + triangle_index * theta_step
            phi = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)

        # 2) the middle 10 triangles
        # 2-0) middle-up triangles
        if 5 <= triangle_index <= 9:
            triangle_index_temp = triangle_index - 5
            theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
            phi = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)

        # 2-1) the middle-down triangles
        if 10 <= triangle_index <= 14:
            triangle_index_temp = triangle_index - 10
            theta = - np.pi + triangle_index_temp * theta_step
            phi = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))

        # 3) the down 5 triangles
        if 15 <= triangle_index <= 19:
            triangle_index_temp = triangle_index - 15
            theta = - np.pi + triangle_index_temp * theta_step
            phi = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))

        thetas.append(theta)
        phis.append(phi)

    return np.array(thetas), np.array(phis)


def pad_pano(pano, padding):
    if padding <= 0:
        return pano

    if pano.ndim == 5:
        b, m = pano.shape[:2]
        pano_pad = rearrange(pano, 'b m c h w -> (b m c) h w')
    elif pano.ndim == 4:
        b = pano.shape[0]
        pano_pad = rearrange(pano, 'b c h w -> (b c) h w')
    else:
        raise NotImplementedError('pano should be 4 or 5 dim')

    pano_pad = F.pad(pano_pad, [padding, ] * 2, mode='circular')
    # pano_pad = torch.cat([
    #     pano_pad[..., :padding, :].flip((-1, -2)),
    #     pano_pad,
    #     pano_pad[..., -padding:, :].flip((-1, -2)),
    # ], dim=-2)

    if pano.ndim == 5:
        pano_pad = rearrange(pano_pad, '(b m c) h w -> b m c h w', b=b, m=m)
    elif pano.ndim == 4:
        pano_pad = rearrange(pano_pad, '(b c) h w -> b c h w', b=b)

    return pano_pad


def unpad_pano(pano_pad, padding):
    if padding <= 0:
        return pano_pad
    return pano_pad[..., padding:-padding]


class Cubemap:
    def __init__(self, cubemap, cube_format):
        if cube_format == 'horizon':
            pass
        elif cube_format == 'list':
            cubemap = py360convert.cube_list2h(cubemap)
        elif cube_format == 'dict':
            cubemap = py360convert.cube_dict2h(cubemap)
        elif cube_format == 'dice':
            cubemap = py360convert.cube_dice2h(cubemap)
        else:
            raise NotImplementedError('unknown cube_format')
        assert len(cubemap.shape) == 3
        assert cubemap.shape[0] * 6 == cubemap.shape[1]
        self.cubemap = cubemap

    def to_equirectangular(self, h, w, mode='bilinear'):
        return Equirectangular(py360convert.c2e(self.cubemap, h, w, mode, cube_format='horizon'))

    @classmethod
    def from_mp3d_skybox(cls, mp3d_skybox_path, scene, view):
        keys = ['U', 'L', 'F', 'R', 'B', 'D']
        images = {}
        for idx, key in enumerate(keys):
            img_path = os.path.join(mp3d_skybox_path, scene, 'matterport_skybox_images', f"{view}_skybox{idx}_sami.jpg")
            images[key] = np.array(Image.open(img_path))
        images['R'] = np.flip(images['R'], 1)
        images['B'] = np.flip(images['B'], 1)
        images['U'] = np.flip(images['U'], 0)
        images['U'] = np.rot90(images['U'], 1)
        images['D'] = np.rot90(images['D'], 1)
        return cls(images, 'dict')

def parse_camera_params(filename: str) -> dict:
    with open(filename, 'r') as f:
        paramdict = {}
        while True: 
            line = f.readline() 
            if not line: 
                break
            lineparts = line.split(" ")
            if lineparts[0]=="scan":
                (loc,row,ori) = lineparts[1].split("_", 3)
                rowid = int(row[1:])
                ori, _ = os.path.splitext(ori)
                oriid = int(ori)                
                trmatrix = np.array([
                    [float(lineparts[3]), float(lineparts[4]), float(lineparts[5]), float(lineparts[6])],
                    [float(lineparts[7]), float(lineparts[8]), float(lineparts[9]), float(lineparts[10])],
                    [float(lineparts[11]), float(lineparts[12]), float(lineparts[13]), float(lineparts[14])],
                    [float(lineparts[15]), float(lineparts[16]), float(lineparts[17]), float(lineparts[18])]
                ])
                if not(loc in paramdict.keys()):
                    paramdict[loc] = {}
                if not(rowid in paramdict[loc].keys()):
                    paramdict[loc][rowid] = {}
                paramdict[loc][rowid][oriid] = trmatrix
    return paramdict


def correct_depth_distortion(depth_img_in):
    depth_img = copy.deepcopy(depth_img_in)
    
    c1 = depth_img.shape[1]/2
    c0 = depth_img.shape[0]/2
    halfFov = createpano.default_fov / 2
    
    for i in range(depth_img.shape[1]):
        angle1 = (abs(i-c1)/c1) * halfFov
        for j in range(depth_img.shape[0]):
            d1 = math.tan(angle1) * depth_img[j,i,0]
            angle0 = (abs(j-c0)/c0) * halfFov    
            d0 = math.tan(angle0) * depth_img[j,i,0] 
            diag = math.sqrt(d0*d0 + d1*d1)
            distToCenter = math.sqrt(float(depth_img[j,i,0])*float(depth_img[j,i,0]) + diag*diag)
            corr = distToCenter - depth_img[j,i,0]
            
            if depth_img[j,i,0] >0:
                result = depth_img[j,i,0] + corr
                if result>65535:
                    result = 65535
                depth_img[j,i,0] = result

        
    return depth_img

class Equirectangular:
    def __init__(self, equirectangular):
        self.equirectangular = equirectangular

    def to_cubemap(self, face_w=256, mode='bilinear'):
        return Cubemap(py360convert.e2c(self.equirectangular, face_w, mode, cube_format='horizon'), 'horizon')

    @classmethod
    def from_file(cls, img_path):
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert("RGB")
        return cls(np.array(img))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(self.equirectangular.astype(np.uint8)).save(path)

    def to_perspective(self, fov, yaw, pitch, hw, mode='bilinear'):
        return py360convert.e2p(self.equirectangular, fov, yaw, pitch, hw, mode=mode)

    def rotate(self, degree):
        if degree % 360 == 0:
            return
        self.equirectangular = np.roll(
            self.equirectangular, int(degree / 360 * self.equirectangular.shape[1]), axis=1)

    def flip(self, flip=True):
        if flip:
            self.equirectangular = np.flip(self.equirectangular, 1)

    @classmethod
    def from_matterport_depth(cls, mp3d_data_path, scene, view, target_hw=(1024, 2048), warp_depth=True, fov_degrees=DEFAULT_FOV_DEGREES):
        """
        Use Matterport3D's undistorted_depth_images and camera_parameters
        to generate a panoramic depth map by inverse projection stitching.

        Args:
            mp3d_data_path: Matterport3D data root directory (contains v1/scans/...)
            scene: scene ID
            view: view/full-view UUID
            target_hw: output panoramic image height and width (H, W)
            warp_depth: whether to apply depth distortion correction (convert depth to radial distance)
            fov_degrees: input depth view horizontal field of view (degrees)

        Returns:
            an Equirectangular instance, containing the stitched uint16 depth map, or None if failed.
        """
        depth_dir = os.path.join(mp3d_data_path, scene, 'undistorted_depth_images')
        params_file = os.path.join(mp3d_data_path,scene, 'undistorted_camera_parameters', f"{scene}.conf")

        if not os.path.isdir(depth_dir):
            print(f"Error: Depth directory not found: {depth_dir}")
            return None
        if not os.path.isfile(params_file):
            print(f"Error: Camera parameters file not found: {params_file}")
            return None

        # 1. load camera parameters
        # assume parse_camera_params returns {view_id: {row_id: {ori_id: transform_matrix}}}
        all_params_nested = parse_camera_params(params_file)
        if all_params_nested is None:
             print(f"Error: Could not load camera parameters from {params_file}")
             return None

        # 2. extract parameters for specific view
        # file format may be scene/scan_id/undistorted...
        # parse_camera_params key may be scan_id, or view_id
        # need to determine which key view parameter corresponds to in paramdict
        # assume view parameter is the first layer key in paramdict
        if view not in all_params_nested:
             # 尝试 view 是否是 scene/view 格式中的 view 部分
             potential_scan_id = view.split('_')[0] # 尝试从 view id 中提取 scan id
             if potential_scan_id in all_params_nested:
                 print(f"Warning: View '{view}' not found directly in params keys. Trying scan_id '{potential_scan_id}'.")
                 # here we need to know which sub-structure view corresponds to in paramdict[scan_id]
                 # assume view ID itself uniquely identifies the position, no scan_id prefix
                 # if the structure returned by parse_camera_params is {loc: {rowid: {oriid: matrix}}}
                 # and view is loc, then the logic below should be correct
                 if view in all_params_nested:
                      view_params = all_params_nested[view]
                 else:
                      print(f"Error: Cannot find parameters for view '{view}' or potential scan_id '{potential_scan_id}' in {params_file}")
                      return None
             else:
                 print(f"Error: View '{view}' not found in camera parameters keys: {list(all_params_nested.keys())}")
                 return None
        else:
             view_params = all_params_nested[view] # view_params is dict {row_id: {ori_id: matrix}}


        # 3. load and process all related depth views
        depth_views = [] # store loaded and possibly corrected depth maps (float32, H, W)
        view_transforms = [] # store corresponding camera-to-world transformation matrices
        print(f"Loading depth views for {scene}/{view}...")
        # --- determine depth map filename format ---
        first_row_id = next(iter(view_params), None)
        if first_row_id is None:
             print(f"Error: No camera parameters found for view {view} after parsing.")
             return None
        first_ori_id = next(iter(view_params[first_row_id]), None)
        if first_ori_id is None:
             print(f"Error: No orientation found for row {first_row_id} in view {view}.")
             return None

        # try different filename formats
        potential_fmt1 = os.path.join(depth_dir, f"{view}_d{first_row_id}_{first_ori_id}.png") # Matterport original format?
        conf_example_name = f"{view}_r{first_row_id}_{first_ori_id}" # corresponds to name in .conf file
        potential_fmt2 = os.path.join(depth_dir, f"{conf_example_name}.png") # official script processed format?
        potential_fmt3 = os.path.join(depth_dir, f"{view}_depth_{first_row_id}_{first_ori_id}.png") # Stanford3dDataset format?

        depth_filename_format = None
        conf_name_template = None # for formatting conf_name
        if os.path.exists(potential_fmt1):
            depth_filename_format = os.path.join(depth_dir, "{view}_d{row_id}_{ori_id}.png")
        elif os.path.exists(potential_fmt2):
            depth_filename_format = os.path.join(depth_dir, "{conf_name}.png")
            conf_name_template = "{view}_r{row_id}_{ori_id}"
        elif os.path.exists(potential_fmt3):
             depth_filename_format = os.path.join(depth_dir, "{view}_depth_{row_id}_{ori_id}.png")
        else:
            print(f"Error: Cannot determine depth filename format. Checked for:")
            print(f"  - {potential_fmt1}")
            print(f"  - {potential_fmt2}")
            print(f"  - {potential_fmt3}")
            # try to list files in directory to guess format
            try:
                example_files = [f for f in os.listdir(depth_dir) if f.startswith(view) and f.endswith('.png')][:5]
                if example_files:
                    print(f"  Example files found in directory: {example_files}")
                else:
                    print(f"  No files starting with '{view}' and ending with '.png' found in {depth_dir}")
            except OSError as e:
                print(f"  Could not list directory {depth_dir}: {e}")
            return None
        print(f"Using depth filename format: {depth_filename_format}")

        loaded_count = 0
        for row_id, orientations in view_params.items():
            for ori_id, transform in orientations.items():
                depth_path = None
                if conf_name_template:
                     conf_name = conf_name_template.format(view=view, row_id=row_id, ori_id=ori_id)
                     depth_path = depth_filename_format.format(conf_name=conf_name)
                else:
                     depth_path = depth_filename_format.format(view=view, row_id=row_id, ori_id=ori_id)

                if os.path.isfile(depth_path):
                    try:
                        img = Image.open(depth_path)
                        depth_array = None # store uint16 (H, W) array
                        # confirm it is a single channel 16-bit image (mode 'I;16' or 'I')
                        if img.mode == 'I;16':
                            depth_array = np.array(img, dtype=np.uint16)
                        elif img.mode == 'I': # 32-bit integer, also possibly a depth map
                             temp_array = np.array(img, dtype=np.int32)
                             # check value range, if suitable for uint16 then convert
                             if np.all(temp_array >= 0) and np.all(temp_array <= 65535):
                                 depth_array = temp_array.astype(np.uint16)
                             else:
                                 print(f"Warning: Depth image {os.path.basename(depth_path)} is 32-bit with values outside uint16 range. Skipping.")
                                 continue
                        elif img.mode == 'L': # 8-bit grayscale image? not likely a depth map
                             print(f"Warning: Depth image {os.path.basename(depth_path)} is 8-bit grayscale (mode 'L'). Skipping.")
                             continue
                        else:
                             print(f"Warning: Depth image {os.path.basename(depth_path)} has unexpected mode {img.mode}. Trying to convert to 'I;16'.")
                             try:
                                 # try to convert, if failed then skip
                                 img_converted = img.convert('I;16')
                                 depth_array = np.array(img_converted, dtype=np.uint16)
                             except Exception as conv_e:
                                 print(f"Failed to convert {os.path.basename(depth_path)} to 'I;16': {conv_e}. Skipping.")
                                 continue

                        if depth_array is None: # if conversion failed or mode not supported
                            continue
                        if depth_array.ndim != 2: # ensure it is 2D
                            print(f"Warning: Loaded depth array from {os.path.basename(depth_path)} has unexpected shape {depth_array.shape}. Skipping.")
                            continue

                        # --- depth processing ---
                        processed_depth_2d = None # store (H, W) float32 depth/radial distance
                        if warp_depth:
                            # correct_depth_distortion needs (H, W, 1) uint16 input
                            # (official code seems to directly pass uint16, internal processing)
                            # we pass uint16 (H, W, 1)
                            depth_array_3d = depth_array[:, :, np.newaxis]

                            # correct_depth_distortion returns float32 radial distance (H, W, 1)
                            corrected_depth_3d = correct_depth_distortion(depth_array_3d)

                            if corrected_depth_3d is not None:
                                # remove extra dimension, store (H, W) float32
                                processed_depth_2d = corrected_depth_3d.squeeze(axis=-1)
                            else:
                                print(f"Warning: Failed to correct depth for {os.path.basename(depth_path)}. Using original depth.")
                                processed_depth_2d = depth_array.astype(np.float32) # use original depth as float32
                        else:
                            # no correction, use original uint16 depth, but convert to float32
                            processed_depth_2d = depth_array.astype(np.float32)

                        if processed_depth_2d is not None:
                            # 再次确认是 2D (H, W)
                            if processed_depth_2d.ndim != 2:
                                print(f"Warning: Processed depth for {os.path.basename(depth_path)} has unexpected shape {processed_depth_2d.shape}. Skipping.")
                                continue

                            depth_views.append(processed_depth_2d) # add (H, W) float32 array
                            view_transforms.append(transform)
                            loaded_count += 1

                    except Exception as e:
                        print(f"Warning: Failed to load or process depth image {os.path.basename(depth_path)}: {e}")
                # else:
                #      print(f"Debug: Depth file not found: {depth_path}") # 

        if not depth_views:
            print(f"Error: No valid depth views could be loaded for {scene}/{view}.")
            return None

        print(f"Loaded {loaded_count} depth views. Starting stitching...")


        h_out, w_out = target_hw
   
        min_depth_map = np.full((h_out, w_out), 65535.0, dtype=np.float32) # 


        phi_coords = np.linspace(-np.pi / 2, np.pi / 2, h_out)
        theta_coords = np.linspace(-np.pi, np.pi, w_out)
        theta_grid, phi_grid = np.meshgrid(theta_coords, phi_coords)

 

        # x = cos(phi) * cos(theta)
        # y = cos(phi) * sin(theta)
        # z = sin(phi)
        world_x = np.cos(phi_grid) * np.cos(theta_grid)
        world_y = np.cos(phi_grid) * np.sin(theta_grid)
        world_z = np.sin(phi_grid)
        world_dirs = np.stack([world_x, world_y, world_z], axis=-1) # Shape: (H_out, W_out, 3)

        # 遍历每个加载的深度视图
        for i, (depth_view, transform) in enumerate(tqdm(zip(depth_views, view_transforms), total=len(depth_views), desc="Projecting views")):


            h_in, w_in = depth_view.shape 
 
            cx = (w_in - 1) / 2.0
            cy = (h_in - 1) / 2.0

            half_fov_rad = np.radians(fov_degrees / 2.0)

            tan_half_fov = np.tan(half_fov_rad)
            if abs(tan_half_fov) < 1e-6:
                 print(f"Warning: Skipping view {i} due to near-zero tan(FOV/2). FOV={fov_degrees}")
                 continue
            fx = cx / tan_half_fov
            fy = fx

            try:
    
                world_to_cam = np.linalg.inv(transform)
            except np.linalg.LinAlgError:
                print(f"Warning: Skipping view {i} due to non-invertible transform matrix.")
                continue

            cam_rotation = world_to_cam[:3, :3]
            # cam_translation = world_to_cam[:3, 3] 

            # cam_dirs = R_world_to_cam @ world_dirs
            world_dirs_flat = world_dirs.reshape(-1, 3) # (H_out*W_out, 3)
            # (3, 3) @ (3, H_out*W_out) -> (3, H_out*W_out) -> transpose -> (H_out*W_out, 3)
            cam_dirs_flat = (cam_rotation @ world_dirs_flat.T).T
            cam_dirs = cam_dirs_flat.reshape(h_out, w_out, 3)

            # 相机坐标系约定: 假设 OpenCV 约定 (+X 右, +Y 下, +Z 前)
            cam_x, cam_y, cam_z = cam_dirs[..., 0], cam_dirs[..., 1], cam_dirs[..., 2]

  
            valid_mask = cam_z > 1e-5 # Z > 0 

            # calculate pixel coordinates projected onto image plane (x, y)
            # x = fx * (cam_x / cam_z) + cx
            # y = fy * (cam_y / cam_z) + cy
            # only calculate for valid directions
            proj_x = np.full_like(cam_x, -1.0) # initialize to invalid value
            proj_y = np.full_like(cam_y, -1.0)

            # avoid division by zero
            safe_cam_z = np.where(valid_mask, cam_z, 1.0) # use 1 to replace invalid Z values to avoid division by zero warning
            proj_x[valid_mask] = fx * (cam_x[valid_mask] / safe_cam_z[valid_mask]) + cx
            proj_y[valid_mask] = fy * (cam_y[valid_mask] / safe_cam_z[valid_mask]) + cy

            # 使用整数索引进行查找 (最近邻)
            proj_x_idx = np.round(proj_x).astype(int)
            proj_y_idx = np.round(proj_y).astype(int)

            # 检查投影坐标是否在输入深度图的边界内
            bounds_mask = (proj_x_idx >= 0) & (proj_x_idx < w_in) & \
                          (proj_y_idx >= 0) & (proj_y_idx < h_in)

            # combine direction validity and boundary validity
            final_valid_mask = valid_mask & bounds_mask

            # get valid panorama pixel coordinates (flattened index)
            valid_pano_indices_flat = np.where(final_valid_mask.flatten())[0]
            if valid_pano_indices_flat.size == 0: # if no valid pixels, skip this view
                continue

            # get corresponding valid projected coordinates (for lookup in input depth map)
            valid_proj_x_idx = proj_x_idx[final_valid_mask]
            valid_proj_y_idx = proj_y_idx[final_valid_mask]

            # lookup depth/radial distance values from input depth map (unit: mm, float32)
            # depth_view is (H_in, W_in) float32
            depth_values = depth_view[valid_proj_y_idx, valid_proj_x_idx]

            # filter out invalid depth values (e.g. 0 or negative)
            # even after correction, original invalid depth (0) should also be filtered
            valid_depth_mask = depth_values > 0.1 # use a small positive threshold to filter invalid depth

            valid_pano_indices_final = valid_pano_indices_flat[valid_depth_mask]
            if valid_pano_indices_final.size == 0: # if no valid depth after filtering, skip
                continue
            valid_depth_values = depth_values[valid_depth_mask]

            # update minimum depth map
            # get current minimum depth of these panorama pixels
            current_min_depths = min_depth_map.flat[valid_pano_indices_final]
            # find which new depth values are smaller than current minimum
            update_mask = valid_depth_values < current_min_depths
            # update depth of these pixels
            min_depth_map.flat[valid_pano_indices_final[update_mask]] = valid_depth_values[update_mask]

        # set pixels not covered by any view (still initial large value) to 0
        min_depth_map[min_depth_map >= 65534.0] = 0 # use a slightly smaller threshold to avoid floating point error

        # convert final depth map (float, mm/radial distance) to uint16 and save
        # note: if warp_depth=True, here saved is uint16 representation of radial distance
        stitched_depth_uint16 = np.round(min_depth_map).clip(0, 65535).astype(np.uint16)

        print("Stitching complete.")
        return cls(stitched_depth_uint16) # return instance containing uint16 depth map

    # ... existing code ...