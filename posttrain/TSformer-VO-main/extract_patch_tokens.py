import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from build_model import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Extract Patch Embedding Tokens from TSformer")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing image frames")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output .npy file")
    parser.add_argument("--window_size", type=int, default=2, help="Number of frames per window (T)")
    parser.add_argument("--image_height", type=int, default=192, help="Input image height")
    parser.add_argument("--image_width", type=int, default=640, help="Input image width")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    return parser.parse_args()

class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, window_size, transform=None):
        self.data_path = data_path
        self.window_size = window_size
        self.transform = transform
        self.is_npy = False
        
        if os.path.isfile(data_path) and data_path.endswith('.npy'):
            self.is_npy = True
            print(f"Loading data from {data_path}...")
            # Expecting (N, H, W, C) or (N, C, H, W)
            self.data = np.load(data_path)
            print(f"Loaded data shape: {self.data.shape}")
            self.num_frames = self.data.shape[0]
        elif os.path.isdir(data_path):
            # Load images sorted
            self.image_paths = sorted(glob.glob(os.path.join(data_path, "*.jpg")) + 
                                      glob.glob(os.path.join(data_path, "*.png")))
            if len(self.image_paths) == 0:
                raise ValueError(f"No images found in {data_path}")
            self.num_frames = len(self.image_paths)
        else:
            raise ValueError(f"Invalid data path: {data_path}")
            
        # Create windows (stride=1)
        self.indices = []
        for i in range(self.num_frames - window_size + 1):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_frame = self.indices[idx]
        imgs = []
        
        for i in range(self.window_size):
            curr_idx = start_frame + i
            
            if self.is_npy:
                # Handle npy frame
                frame = self.data[curr_idx]
                # Assuming frame is (H, W, C) uint8 or float
                if frame.ndim == 3:
                    if frame.shape[0] <= 4: # Likely (C, H, W)
                         # Convert to (H, W, C) for PIL compatibility if needed, or just tensor
                         frame = frame.transpose(1, 2, 0)
                    
                    # Convert to PIL for consistent transform
                    if frame.dtype != np.uint8:
                         if frame.max() <= 1.0:
                             frame = (frame * 255).astype(np.uint8)
                         else:
                             frame = frame.astype(np.uint8)
                    
                    img = Image.fromarray(frame)
                else:
                    raise ValueError(f"Unexpected frame shape: {frame.shape}")
            else:
                # Handle image file
                img = Image.open(self.image_paths[curr_idx]).convert("RGB")
                
            if self.transform:
                img = self.transform(img)
            imgs.append(img.unsqueeze(0)) # (1, C, H, W)
        
        # Stack to (C, T, H, W) matching model input expectation
        imgs = torch.cat(imgs, dim=0) # (T, C, H, W)
        imgs = imgs.transpose(0, 1)   # (C, T, H, W)
        return imgs

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Model Parameters
    # Matching the 'small' config from train.py by default
    model_params = {
        "dim": args.embed_dim,
        "image_size": (args.image_height, args.image_width),
        "patch_size": args.patch_size,
        "attention_type": 'divided_space_time',
        "num_frames": args.window_size,
        "num_classes": 0, # Not needed for feature extraction
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "time_only": False,
    }
    
    # Mock args for build_model
    build_args = {
        "checkpoint_path": "dummy",
        "checkpoint": None, # We load manually if needed
        "pretrained_ViT": False,
        "epoch_init": 0,
        "best_val": 0
    }

    print("Building model...")
    # build_model returns (model, args)
    model, _ = build_model(build_args, model_params)
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # Handle if checkpoint has 'model_state_dict' or is just the dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Remove head keys if mismatch (since we set num_classes=0)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load status: {msg}")

    model = model.to(device)
    model.eval()

    # 2. Setup Data
    preprocess = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # Using ImageNet standard as in train_custom.py
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    dataset = SimpleImageDataset(args.data_dir, args.window_size, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Processing {len(dataset)} windows...")

    all_tokens = []

    # 3. Extraction Loop
    with torch.no_grad():
        for batch_idx, images in enumerate(tqdm(dataloader)):
            # images shape: (B, C, T, H, W)
            images = images.to(device)
            
            # Extract Patch Embeddings
            # We access the patch_embed layer directly from the VisionTransformer
            # model is an instance of VisionTransformer (from build_model)
            # Or if distributed, it might be wrapped. build_model returns VisionTransformer directly usually.
            
            # Check if model is wrapped (e.g. DDP) - usually not here
            patch_embed_layer = model.patch_embed
            
            # Forward pass through patch_embed
            # The forward method of PatchEmbed expects (B, C, T, H, W)
            # And returns (x, T, W) where x is (B*T, N, C)
            tokens, _, _ = patch_embed_layer(images)
            
            # tokens shape: (B*T, Num_Patches, Embed_Dim)
            all_tokens.append(tokens.cpu().numpy())

    # 4. Save
    if len(all_tokens) > 0:
        all_tokens = np.concatenate(all_tokens, axis=0)
        print(f"Extracted tokens shape: {all_tokens.shape}")
        print(f"Saving to {args.output_path}...")
        np.save(args.output_path, all_tokens)
        print("Done.")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()
