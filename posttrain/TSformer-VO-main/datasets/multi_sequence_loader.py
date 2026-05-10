import json
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class MultiSequenceDataset(Dataset):
    """
    Dataset for loading VO data from multiple trajectory sequences.
    
    Structure Assumption:
    1. poses_dir: Contains JSON files (e.g., 'traj_001.json', 'traj_002.json')
    2. images_dir: Contains subfolders with matching names (e.g., 'traj_001/', 'traj_002/')
       Inside subfolders: Sequential images (e.g., '0.jpg', '1.jpg'...)
       
    Arguments:
        poses_dir {str}: path to the folder containing all JSON pose files
        images_dir {str}: path to the folder containing all image subfolders
        window_size {int}: number of frames in a window (default 2)
        transform {callable}: transform to apply to images
        img_extension {str}: extension of image files (default .jpg)
        max_sequences {int}: limit number of sequences to load (for debugging/testing)
    """

    def __init__(self,
                 poses_dir,
                 images_dir,
                 window_size=2,
                 transform=None,
                 img_extension=".jpg",
                 max_sequences=None
                 ):

        self.poses_dir = poses_dir
        self.images_dir = images_dir
        self.transform = transform
        self.window_size = window_size
        self.img_extension = img_extension
        
        # 1. Find all Pose JSON files
        self.json_files = sorted(glob.glob(os.path.join(poses_dir, "*.json")))
        
        if max_sequences:
            self.json_files = self.json_files[:max_sequences]
            
        if len(self.json_files) == 0:
            raise FileNotFoundError(f"No .json files found in {poses_dir}")
            
        print(f"Found {len(self.json_files)} trajectory sequences.")

        # 2. Build Index and Load Poses
        # We will load all poses into memory (efficient for <10GB datasets)
        # self.samples will store tuples: (image_folder_path, pose_sequence_data, start_frame_index)
        self.samples = [] 
        
        # Temporary lists for calculating statistics
        all_trans = []
        all_rots = []
        
        print("Indexing sequences and loading poses...")
        
        # We'll use a sample for stats calculation to speed up start time if data is huge
        stats_sample_rate = max(1, len(self.json_files) // 500) # Sample ~500 sequences for stats
        
        for idx, json_file in enumerate(tqdm(self.json_files)):
            seq_name = os.path.splitext(os.path.basename(json_file))[0]
            
            # Check corresponding image folder
            img_folder = os.path.join(images_dir, seq_name)
            if not os.path.isdir(img_folder):
                # Try checking if images are in root (if user put all images in one folder? unlikely for 20k seqs)
                # Assuming standard structure: images_dir/seq_name/
                # print(f"Warning: Image folder for {seq_name} not found at {img_folder}. Skipping.")
                continue

            # Load Poses
            try:
                with open(json_file, 'r') as f:
                    poses = json.load(f)
                poses = np.array(poses, dtype=np.float32) # Shape: (N, 6)
            except Exception as e:
                print(f"Error reading {json_file}: {e}. Skipping.")
                continue
                
            num_frames = len(poses)
            if num_frames < window_size:
                continue

            # Add to stats if sampled
            if idx % stats_sample_rate == 0:
                # Skip first frame (0,0,0,0,0,0) usually
                if len(poses) > 1:
                    all_trans.append(poses[1:, 0:3])
                    all_rots.append(poses[1:, 3:6])
            
            # Create Windows
            # For a sequence of length N, we can create (N - window_size + 1) windows
            # We store: (img_folder, poses_array_for_this_seq, start_index)
            # Optimization: Don't store full poses array in every sample. 
            # Store index to a list of poses arrays.
            
            # But since we are creating a list of samples, we need to be careful with memory.
            # Storing the full numpy array 'poses' in the list `self.samples` multiple times is bad?
            # Actually, Python references the same object. So it's fine.
            
            for i in range(num_frames - window_size + 1):
                self.samples.append({
                    "img_folder": img_folder,
                    "poses": poses,         # Reference to the numpy array
                    "start_idx": i
                })

        print(f"Total valid samples (windows) generated: {len(self.samples)}")
        
        # 3. Compute Normalization Stats
        print("Computing normalization stats...")
        if len(all_trans) > 0:
            all_trans = np.concatenate(all_trans, axis=0)
            all_rots = np.concatenate(all_rots, axis=0)
            
            self.mean_t = np.mean(all_trans, axis=0)
            self.std_t = np.std(all_trans, axis=0) + 1e-6
            
            self.mean_angles = np.mean(all_rots, axis=0)
            self.std_angles = np.std(all_rots, axis=0) + 1e-6
        else:
            # Fallback (should not happen if data exists)
            self.mean_t = np.zeros(3)
            self.std_t = np.ones(3)
            self.mean_angles = np.zeros(3)
            self.std_angles = np.ones(3)
            
        print(f"Mean T: {self.mean_t}")
        print(f"Mean R: {self.mean_angles}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            imgs {tensor}: (C, T, H, W) normalized images
            y {tensor}: flattened array of normalized relative poses [Rx, Ry, Rz, Tx, Ty, Tz]
        """
        sample = self.samples[idx]
        img_folder = sample["img_folder"]
        poses = sample["poses"]
        start_frame = sample["start_idx"]
        
        # 1. Load Images
        imgs = []
        for i in range(self.window_size):
            frame_idx = start_frame + i
            
            # Try different filename formats
            # Priority 1: 0.jpg, 1.jpg
            img_name_simple = f"{frame_idx}{self.img_extension}"
            img_path = os.path.join(img_folder, img_name_simple)
            
            # Priority 2: 000000.jpg (KITTI style)
            if not os.path.exists(img_path):
                img_name_padded = f"{frame_idx:06d}{self.img_extension}"
                img_path = os.path.join(img_folder, img_name_padded)
                
            # If still not found, try to list dir and pick by index (Slow, but robust)
            # We assume filenames are predictable for performance. 
            
            if not os.path.exists(img_path):
                # Fallback: Maybe user has frame_1.jpg?
                # We raise error to force user to fix naming, otherwise training is too slow
                raise FileNotFoundError(f"Image for frame {frame_idx} not found in {img_folder}")

            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0)) # Add time dimension (1, C, H, W)
            except OSError:
                # Handle corrupted images
                print(f"Warning: Corrupted image {img_path}. Returning zeros.")
                imgs.append(torch.zeros(1, 3, 192, 640)) # Assuming default size

        # Concatenate along time dimension: (T, C, H, W)
        imgs = torch.cat(imgs, dim=0)
        
        # Transpose to (C, T, H, W) as expected by the model
        imgs = imgs.transpose(0, 1)

        # 2. Process Labels
        # We need relative poses for the frames in the window (excluding the first one relative to prev)
        # The window is [t, t+1, ...]. We need poses at t+1, t+2... relative to their prev.
        # Our 'poses' array usually contains relative pose at index i (motion from i-1 to i).
        
        y = []
        for i in range(1, self.window_size):
            pose_idx = start_frame + i
            if pose_idx < len(poses):
                pose_raw = poses[pose_idx]
                
                # Extract T and R (Assuming Input is [Tx, Ty, Tz, Rx, Ry, Rz])
                t_raw = pose_raw[0:3]
                r_raw = pose_raw[3:6]
                
                # Normalize
                t_norm = (t_raw - self.mean_t) / self.std_t
                r_norm = (r_raw - self.mean_angles) / self.std_angles
                
                # Output [R, T] (Angles first)
                y.extend(list(r_norm) + list(t_norm))
            else:
                # Should not happen due to __init__ checks
                y.extend([0]*6)
            
        y = np.array(y, dtype=np.float32)
        
        return imgs, y
