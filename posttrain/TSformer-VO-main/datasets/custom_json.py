import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomJsonDataset(Dataset):
    """
    Dataset for loading VO data from JSON logs and an image folder.
    
    Arguments:
        json_path {str}: path to the preprocessed_logs.json (relative poses)
        image_dir {str}: path to the folder containing images
        window_size {int}: number of frames in a window (default 2)
        transform {callable}: transform to apply to images
        img_extension {str}: extension of image files (default .jpg)
    """

    def __init__(self,
                 json_path,
                 image_dir,
                 window_size=2,
                 transform=None,
                 img_extension=".jpg"
                 ):

        self.image_dir = image_dir
        self.transform = transform
        self.window_size = window_size
        self.img_extension = img_extension

        # Load JSON data
        # Expecting format: List of [tx, ty, tz, rx, ry, rz]
        # Frame 0 is usually all zeros (start point)
        with open(json_path, 'r') as f:
            self.poses = json.load(f)
        
        # Convert to numpy array
        self.poses = np.array(self.poses) # Shape: (N, 6)
        
        # Data Validation
        print(f"Loaded {len(self.poses)} poses from {json_path}")
        
        # Calculate Normalization Statistics
        # We skip the first frame (index 0) because it's usually 0 motion or undefined
        valid_poses = self.poses[1:] 
        
        # User data appears to be [Tx, Ty, Tz, Rx, Ry, Rz] based on magnitude
        # Model expects [Rx, Ry, Rz, Tx, Ty, Tz]
        # We need to swap and then calculate stats
        
        # Split Translation and Rotation
        self.raw_trans = valid_poses[:, 0:3]
        self.raw_rot = valid_poses[:, 3:6]
        
        # Calculate mean and std for normalization
        self.mean_t = np.mean(self.raw_trans, axis=0)
        self.std_t = np.std(self.raw_trans, axis=0) + 1e-6
        
        self.mean_angles = np.mean(self.raw_rot, axis=0)
        self.std_angles = np.std(self.raw_rot, axis=0) + 1e-6
        
        print("Normalization Stats Computed:")
        print(f"Mean T: {self.mean_t}, Std T: {self.std_t}")
        print(f"Mean R: {self.mean_angles}, Std R: {self.std_angles}")

        # Create Windows
        # A window of size 2 (frames i, i+1) predicts pose i+1 (relative to i)
        self.windows = []
        num_frames = len(self.poses)
        
        # Ensure we have enough frames for at least one window
        if num_frames >= window_size:
            # We can start windows from frame 0 up to num_frames - window_size
            for i in range(num_frames - window_size + 1):
                self.windows.append(i)
        else:
            print("Warning: Not enough frames for the specified window size.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Returns:
            imgs {tensor}: (C, T, H, W) normalized images
            y {tensor}: flattened array of normalized relative poses [Rx, Ry, Rz, Tx, Ty, Tz]
        """
        start_frame = self.windows[idx]
        
        # 1. Load Images
        imgs = []
        for i in range(self.window_size):
            frame_idx = start_frame + i
            
            # Try different filename formats
            # Priority 1: 0.jpg, 1.jpg
            img_name_simple = f"{frame_idx}{self.img_extension}"
            img_path = os.path.join(self.image_dir, img_name_simple)
            
            # Priority 2: 000000.jpg (KITTI style)
            if not os.path.exists(img_path):
                img_name_padded = f"{frame_idx:06d}{self.img_extension}"
                img_path = os.path.join(self.image_dir, img_name_padded)
                
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image for frame {frame_idx} not found at {img_path} or simple format.")

            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img.unsqueeze(0)) # Add time dimension (1, C, H, W)
            
        # Concatenate along time dimension: (T, C, H, W)
        imgs = torch.cat(imgs, dim=0)
        
        # Transpose to (C, T, H, W) as expected by the model
        imgs = imgs.transpose(0, 1)

        # 2. Load and Normalize Labels
        # For a window [t, t+1, t+2], we need relative poses at t+1 (from t) and t+2 (from t+1)
        # Our poses array index i contains motion from i-1 to i.
        # So for window starting at `start_frame`, the relevant pose indices are `start_frame + 1` to `start_frame + window_size - 1`
        
        y = []
        for i in range(1, self.window_size):
            pose_idx = start_frame + i
            pose_raw = self.poses[pose_idx]
            
            # Extract T and R
            # Assuming input is [Tx, Ty, Tz, Rx, Ry, Rz]
            t_raw = pose_raw[0:3]
            r_raw = pose_raw[3:6]
            
            # Normalize
            t_norm = (t_raw - self.mean_t) / self.std_t
            r_norm = (r_raw - self.mean_angles) / self.std_angles
            
            # Concatenate as [R, T] (Angles first, then Translation) - Matches KITTI.py logic
            # KITTI.py line 113: y.append(list(angles) + list(t))
            y.extend(list(r_norm) + list(t_norm))
            
        y = np.array(y, dtype=np.float32)
        
        return imgs, y
