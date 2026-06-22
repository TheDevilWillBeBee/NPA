import os
import json
import numpy as np
import torch
from PIL import Image

def blender_to_colmap(view_matrix):
    """
    Convert view matrices from Blender to COLMAP convention.
    
    Args:
        view_matrix: torch.Tensor of shape [N, 4, 4] (world-to-camera matrices)
    
    Returns:
        Converted view matrices in COLMAP convention
    """
    # Transformation: flip Y and Z axes
    # Blender: +X right, +Y up, -Z forward
    # COLMAP: +X right, -Y up, +Z forward
    
    flip = torch.tensor([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=view_matrix.dtype, device=view_matrix.device)
    
    # Apply transformation in camera space
    view_matrix_colmap = flip @ view_matrix
    
    return view_matrix_colmap

def get_dataset(rootdir, train=True, resize=None, device='cpu'):
    """
    Load NeRF Synthetic dataset.
    
    Args:
        rootdir: Path to scene directory (e.g., 'data/nerf_synthetic/lego')
        train: If True, load training set; otherwise load validation set
        resize: Optional tuple (new_height, new_width) to resize images
    
    Returns:
        Dictionary containing:
            - images: RGBA images [C, H, W, 4]
            - viewmats: Camera view matrices [C, 4, 4]
            - Ks: Camera intrinsics [C, 3, 3]
    """
    # Determine which split to load
    split = "train" if train else "val"
    transform_file = os.path.join(rootdir, f"transforms_{split}.json")
    
    # Load transform JSON
    with open(transform_file, 'r') as f:
        meta = json.load(f)
    
    # Extract camera parameters
    camera_angle_x = meta['camera_angle_x']
    frames = meta['frames']
    
    # Get original image dimensions from first image
    first_image_path = os.path.join(rootdir, frames[0]['file_path'] + '.png')
    first_img = Image.open(first_image_path)
    orig_width, orig_height = first_img.size
    
    # Determine final dimensions
    if resize is not None:
        height, width = resize
    else:
        width, height = orig_width, orig_height
    
    # Compute focal length from FOV (based on original width)
    # focal = width / (2 * tan(camera_angle_x / 2))
    focal_orig = 0.5 * orig_width / np.tan(0.5 * camera_angle_x)
    
    # Scale focal length if resizing
    focal_x = focal_orig * (width / orig_width)
    focal_y = focal_orig * (height / orig_height)
    
    # Prepare lists to collect data
    images_list = []
    viewmats_list = []
    Ks_list = []
    
    for frame in frames:
        # Load image
        image_path = os.path.join(rootdir, frame['file_path'] + '.png')
        img = Image.open(image_path)
        
        # Resize if needed
        if resize is not None:
            # Use LANCZOS for high-quality downsampling/upsampling
            img = img.resize((width, height), Image.LANCZOS)
        
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Ensure RGBA format
        if img_array.shape[-1] == 3:
            # Add alpha channel if missing
            alpha = np.ones((*img_array.shape[:2], 1), dtype=np.float32)
            img_array = np.concatenate([img_array, alpha], axis=-1)
        
        images_list.append(img_array)
        
        # Extract camera-to-world transform matrix
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        
        # Convert to world-to-camera (view matrix)
        # viewmat = inverse(c2w)
        w2c = np.linalg.inv(c2w)
        viewmats_list.append(w2c)
        
        # Construct intrinsics matrix K
        # K = [[fx, 0, cx],
        #      [0, fy, cy],
        #      [0,  0,  1]]
        K = np.array([
            [focal_x, 0, width / 2.0],
            [0, focal_y, height / 2.0],
            [0, 0, 1]
        ], dtype=np.float32)
        Ks_list.append(K)
    
    # Stack into tensors
    images = torch.from_numpy(np.stack(images_list, axis=0))  # [C, H, W, 4]
    viewmats = torch.from_numpy(np.stack(viewmats_list, axis=0))  # [C, 4, 4]
    Ks = torch.from_numpy(np.stack(Ks_list, axis=0))  # [C, 3, 3]
    
    viewmats = blender_to_colmap(viewmats)

    return {
        "images": images.to(device),
        "viewmats": viewmats.to(device),
        "Ks": Ks.to(device),
    }
