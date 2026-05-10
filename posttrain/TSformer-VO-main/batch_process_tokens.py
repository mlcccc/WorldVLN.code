import argparse
import json
import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch.multiprocessing as mp
import math

# Add project root to path to import model modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_model import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Process Video Tokens from JSONL")
    parser.add_argument("--meta_file", type=str, required=True, help="Path to wan_meta.jsonl")
    parser.add_argument("--image_height", type=int, default=192, help="Input image height")
    parser.add_argument("--image_width", type=int, default=640, help="Input image width")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--checkpoint", type=str, default="/home/dataset-assist-0/xjc/TSformer-VO-main/checkpoint/checkpoint_model3_exp20.pth", help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    return parser.parse_args()

def load_model(args, device):
    model_params = {
        "dim": args.embed_dim,
        "image_size": (args.image_height, args.image_width),
        "patch_size": args.patch_size,
        "attention_type": 'divided_space_time',
        "num_frames": 1, # We process single frames
        "num_classes": 0,
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "time_only": False,
    }
    
    build_args = {
        "checkpoint_path": "dummy",
        "checkpoint": None,
        "pretrained_ViT": False,
        "epoch_init": 0,
        "best_val": 0
    }

    print(f"Building model on {device}...")
    model, _ = build_model(build_args, model_params)
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model

def process_video(video_path, model, preprocess, device, batch_size):
    if not os.path.exists(video_path):
        # print(f"Error: Video not found at {video_path}")
        return

    # Check frame count
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if (total_frames - 1) % 4 != 0:
        # print(f"Warning: Frame count {total_frames} is not 4k+1 for {video_path}")
        pass
    
    # Prepare output directory
    video_dir = os.path.dirname(video_path)
    # Output to reshape_actionhead_data/frame_tokens (user requirement)
    output_dir = os.path.join(video_dir, "frame_tokens")
    os.makedirs(output_dir, exist_ok=True)
    
    # Indices to process: 0, 4, 8, ... (1st, 5th, 9th...)
    indices = range(0, total_frames, 4)
    
    # Check if already processed (simple check, maybe count files?)
    # For now, just overwrite or skip if strict check needed. 
    # We'll proceed to process.

    # Batch processing
    batch_frames = []
    batch_indices = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {idx} from {video_path}")
            continue
            
        # Convert BGR (OpenCV) to RGB (PIL)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # Preprocess
        img_tensor = preprocess(img) # (C, H, W)
        
        batch_frames.append(img_tensor)
        batch_indices.append(idx)
        
        # If batch full or last item, run inference
        if len(batch_frames) >= batch_size or idx == indices[-1]:
            # Stack: (B, C, H, W)
            input_tensor = torch.stack(batch_frames).to(device)
            
            # Reshape for PatchEmbed: (B, C, T=1, H, W)
            input_tensor = input_tensor.unsqueeze(2) 
            
            with torch.no_grad():
                # model.patch_embed returns (x, T, W) -> x is (B*T, N, D)
                tokens, _, _ = model.patch_embed(input_tensor)
                
            # Save individual tokens
            tokens_np = tokens.cpu().numpy() # (B, N, D)
            
            for i, token_data in enumerate(tokens_np):
                frame_idx = batch_indices[i]
                # Naming: token{frame_idx+1}.npy (e.g., token1.npy, token5.npy)
                save_name = f"token{frame_idx+1}.npy"
                save_path = os.path.join(output_dir, save_name)
                np.save(save_path, token_data)
                
            batch_frames = []
            batch_indices = []
            
    cap.release()
    # print(f"Processed {video_path}: {len(indices)} frames saved to {output_dir}")

def worker(rank, args, chunks):
    # Get the chunk for this worker
    lines = chunks[rank]
    
    # Set device for this worker
    device_id = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"Worker {rank} using device {device} processing {len(lines)} videos")
    
    # Load Model
    model = load_model(args, device)
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    base_dir = "/home/dataset-assist-0/xjc"
    
    for line in tqdm(lines, position=rank, desc=f"Worker {rank}"):
        try:
            entry = json.loads(line)
            
            # Construct absolute path logic
            video_rel_path = entry.get('video', '')
            if not video_rel_path:
                continue
            
            # Full path: /home/dataset-assist-0/xjc/uavflowdatasim_output/7406/video.mp4
            video_full_path = os.path.join(base_dir, video_rel_path)
            
            # Target video is in reshape_actionhead_data sibling folder
            video_dir = os.path.dirname(video_full_path)
            target_path = os.path.join(video_dir, "reshape_actionhead_data", "video.mp4")
            
            process_video(target_path, model, preprocess, device, args.batch_size)
            
        except Exception as e:
            print(f"Error processing line: {line[:50]}... -> {e}")

def main():
    args = parse_args()
    
    # Read JSONL
    print(f"Reading meta file: {args.meta_file}")
    with open(args.meta_file, 'r') as f:
        lines = f.readlines()
        
    print(f"Found {len(lines)} entries. Starting multiprocessing on {args.num_gpus} GPUs...")
    
    # Split lines
    chunk_size = math.ceil(len(lines) / args.num_gpus)
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    # Pad chunks if necessary (though mp.spawn limits nprocs)
    
    mp.spawn(worker, args=(args, chunks), nprocs=min(len(chunks), args.num_gpus), join=True)

if __name__ == "__main__":
    main()
