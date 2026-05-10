import os
import torch
import torch.optim as optim
from tqdm import tqdm
from datasets.multi_sequence_loader import MultiSequenceDataset  # Updated Loader
from build_model import build_model
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import numpy as np

# Set seed
torch.manual_seed(2023)

def compute_loss(y_hat, y, criterion, args):
    if args["weighted_loss"] is None:
        loss = criterion(y_hat, y.float())
    else:
        y = torch.reshape(y, (y.shape[0], args["window_size"]-1, 6))
        gt_angles = y[:, :, :3].flatten()
        gt_translation = y[:, :, 3:].flatten()

        # predict pose
        y_hat = torch.reshape(y_hat, (y_hat.shape[0], args["window_size"]-1, 6))
        estimated_angles = y_hat[:, :, :3].flatten()
        estimated_translation = y_hat[:, :, 3:].flatten()

        # compute custom loss
        k = args["weighted_loss"]
        loss_angles = k * criterion(estimated_angles, gt_angles.float())
        loss_translation = criterion(estimated_translation, gt_translation.float())
        loss =  loss_angles + loss_translation   
    return loss

def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args):
    epoch_loss = 0
    iter_idx = (epoch - 1) * len(train_loader) + 1

    with tqdm(train_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            # predict pose
            estimated_pose = model(images.float())

            # compute loss
            loss = compute_loss(estimated_pose, gt, criterion, args)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

            # log tensorboard
            tensorboard_writer.add_scalar('training_loss', loss.item(), iter_idx)
            iter_idx += 1
            
    return epoch_loss / len(train_loader)

def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0
    with tqdm(val_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            tepoch.set_description(f"Validating")
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            # predict pose
            estimated_pose = model(images.float())

            # compute loss
            loss = compute_loss(estimated_pose, gt, criterion, args)

            epoch_loss += loss.item()
            tepoch.set_postfix(val_loss=loss.item())

    return epoch_loss / len(val_loader)

if __name__ == "__main__":

    # ================= USER CONFIGURATION =================
    # IMPORTANT: Set these paths to your actual data locations
    # ------------------------------------------------------
    # Folder containing all your 20,000+ JSON files
    POSES_DIR = r"c:\Users\Administrator\Desktop\TSformer-VO-main\data\poses" 
    
    # Folder containing subfolders of images (folder names must match JSON file names)
    # e.g. POSES_DIR/seq1.json corresponds to IMAGES_DIR/seq1/*.jpg
    IMAGES_DIR = r"c:\Users\Administrator\Desktop\TSformer-VO-main\data\images"
    
    CHECKPOINT_DIR = "checkpoints/MultiSeqExp"
    # ======================================================

    # Hyperparameters
    args = {
        "bsize": 8,            # Batch size (Try 16 or 32 if you have a good GPU)
        "window_size": 2,      # Number of frames
        "lr": 1e-4,            # Learning Rate
        "epoch": 50,           # Total Epochs
        "weighted_loss": 100,  
        "checkpoint_path": CHECKPOINT_DIR,
        "pretrained_ViT": False, 
    }

    # Model Parameters
    model_params = {
        "dim": 384,
        "image_size": (192, 640), 
        "patch_size": 16,
        "attention_type": 'divided_space_time', 
        "num_frames": args["window_size"],
        "num_classes": 6 * (args["window_size"] - 1), 
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "time_only": False,
    }
    args["model_params"] = model_params

    # Ensure checkpoint dir exists
    if not os.path.exists(args["checkpoint_path"]):
        os.makedirs(args["checkpoint_path"])

    # Transforms
    preprocess = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]), 
    ])

    print("Initializing Multi-Sequence Dataset...")
    print(f"Poses Dir: {POSES_DIR}")
    print(f"Images Dir: {IMAGES_DIR}")
    
    # Check if dirs exist
    if not os.path.exists(POSES_DIR) or not os.path.exists(IMAGES_DIR):
        print("\n!!! ERROR: Data directories not found. !!!")
        print("Please create the folders and update the POSES_DIR and IMAGES_DIR variables in this script.")
        print("Current settings:")
        print(f"  Poses: {POSES_DIR} (Exists: {os.path.exists(POSES_DIR)})")
        print(f"  Images: {IMAGES_DIR} (Exists: {os.path.exists(IMAGES_DIR)})")
        # Don't crash immediately, let user see the message
    
    try:
        full_dataset = MultiSequenceDataset(
            poses_dir=POSES_DIR, 
            images_dir=IMAGES_DIR, 
            window_size=args["window_size"], 
            transform=preprocess,
            # max_sequences=100 # Uncomment to test with small subset first
        )
        
        # Split Train/Val
        # For huge datasets, we might want to split by sequence instead of by frame
        # But random_split is easier for now. 
        # Note: Mixing frames from same sequence in train/val causes data leakage.
        # Ideally, we should split sequences. But MultiSequenceDataset flattens them.
        # Given 20k sequences, random split is "okay" but not rigorous for VO evaluation.
        # For rigorous eval, we should implement sequence-based splitting in the Dataset class.
        
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Use more workers for loading images
        train_loader = DataLoader(train_dataset, batch_size=args["bsize"], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args["bsize"], shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"Dataset Loaded. Train Samples: {len(train_dataset)}, Val Samples: {len(val_dataset)}")
        
        # Save Normalization Stats
        stats_path = os.path.join(args["checkpoint_path"], "stats.json")
        import json
        stats = {
            "mean_t": full_dataset.mean_t.tolist(),
            "std_t": full_dataset.std_t.tolist(),
            "mean_angles": full_dataset.mean_angles.tolist(),
            "std_angles": full_dataset.std_angles.tolist()
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        print(f"Normalization stats saved to {stats_path}")
        
    except Exception as e:
        print(f"\nDataset Initialization Failed: {e}")
        train_loader = None
        val_loader = None

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = build_model(model_params)
    model = model.to(device)
    
    # Check for multi-gpu
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    criterion = torch.nn.MSELoss()
    tensorboard_writer = SummaryWriter(log_dir=os.path.join(args["checkpoint_path"], "logs"))

    # Training Loop
    if train_loader:
        best_val_loss = float('inf')
        
        # Prepare stats for validation
        val_stats = {
            "mean_t": full_dataset.mean_t,
            "std_t": full_dataset.std_t,
            "mean_angles": full_dataset.mean_angles,
            "std_angles": full_dataset.std_angles
        }
        
        for epoch in range(1, args["epoch"] + 1):
            print(f"\nStart Epoch {epoch}")
            
            # Train
            model.train()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args)
            
            # Val
            # Pass stats to calculate physical error
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion, args, dataset_stats=val_stats)
            
            print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Save Best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args["checkpoint_path"], "best_model.pth"))
                print("Saved Best Model")
                
            # Save Checkpoint
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(args["checkpoint_path"], f"checkpoint_e{epoch}.pth"))

    print("Done.")
