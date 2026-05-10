import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from latent_patch_embed import LatentToPatchEmbed

class LatentPretrainDataset(Dataset):
    def __init__(self, base_dir, ids):
        self.samples = []
        
        for vid_id in ids:
            npy_path = os.path.join(base_dir, vid_id, "reshape_data_actionhead", "video_summed_codes.npy")
            token_dir = os.path.join(base_dir, vid_id, "reshape_data_actionhead", "frame_tokens")
            
            if not os.path.exists(npy_path) or not os.path.exists(token_dir):
                print(f"Skipping {vid_id}: Data missing")
                continue
                
            try:
                # Shape: (B, C, T, H, W) e.g. (1, 16, 11, 32, 32)
                latents = np.load(npy_path)
                
                # Squeeze batch dim if B=1
                if latents.ndim == 5 and latents.shape[0] == 1:
                    latents = latents[0] # (C, T, H, W)
                
                # Check dims
                if latents.ndim == 4: # (C, T, H, W)
                    # Transpose to (T, C, H, W) for easier iteration
                    latents = latents.transpose(1, 0, 2, 3)
                
                num_frames = latents.shape[0]
                
                # Pair with tokens
                # Logic: Latent t corresponds to token_{1 + t*4}
                for t in range(num_frames):
                    frame_idx = 1 + t * 4
                    token_name = f"token{frame_idx}.npy"
                    token_path = os.path.join(token_dir, token_name)
                    
                    if os.path.exists(token_path):
                        self.samples.append({
                            'latent': latents[t], # (C, H, W)
                            'token_path': token_path
                        })
            except Exception as e:
                print(f"Error loading {vid_id}: {e}")
                
        print(f"Initialized dataset with {len(self.samples)} samples from IDs {ids}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Latent: (C, H, W)
        latent = torch.from_numpy(item['latent']).float()
        
        # Target: (Num_Patches, Embed_Dim)
        target = np.load(item['token_path'])
        target = torch.from_numpy(target).float()
        
        return latent, target

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Model
    model = LatentToPatchEmbed(
        latent_dim=16, 
        embed_dim=384, 
        img_size=(192, 640),
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers
    ).to(device)
    
    # Calculate params
    num_params = count_parameters(model)
    print(f"\nModel Architecture: LatentToPatchEmbed")
    print(f"Total Trainable Parameters: {num_params:,}")
    
    # 2. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 3. Data
    # Hardcoded IDs for testing as per request
    test_ids = ["1", "2", "3"]
    dataset = LatentPretrainDataset(args.data_dir, test_ids)
    
    if len(dataset) == 0:
        print("No data found. Exiting.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    print(f"Start training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for latents, targets in pbar:
            # latents: (B, C, H, W)
            # targets: (B, N, D)
            latents, targets = latents.to(device), targets.to(device)
            
            # Forward
            # Model expects (B, T, C, H, W) or (B, C, H, W)
            # Our batch is (B, C, H, W), so T=1 implicitly
            features, _, _ = model(latents) # (B*1, N, D)
            
            loss = criterion(features, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.save_dir, f"latent_embed_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Base path to uavflowdatasim_output")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the new layer")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of residual layers")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(args)
