import os
import torch
import torch.optim as optim
from tqdm import tqdm
from datasets.custom_json import CustomJsonDataset  # Use our new dataset
from build_model import build_model
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import numpy as np

# Set seed
torch.manual_seed(2023)

def compute_loss(y_hat, y, criterion, args):
    # y_hat: (Batch, Output_Dim)
    # y: (Batch, Output_Dim)
    
    # Reshape to (Batch, Time-1, 6)
    # 6 dims are [Rx, Ry, Rz, Tx, Ty, Tz]
    
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
    # Please update these paths to match your actual file locations
    JSON_PATH = r"c:\Users\Administrator\Desktop\TSformer-VO-main\preprocessed_logs.json"
    IMAGE_DIR = r"c:\Users\Administrator\Desktop\TSformer-VO-main\images" # Update this!
    CHECKPOINT_DIR = "checkpoints/CustomExp"
    # ======================================================

    # Hyperparameters
    args = {
        "bsize": 8,            # Batch size
        "window_size": 2,      # Number of frames (2 frames -> predict 1 motion)
        "lr": 1e-4,            # Learning Rate
        "epoch": 50,           # Total Epochs
        "weighted_loss": 100,  # Weight for rotation loss (usually rotation is small so weight it up)
        "checkpoint_path": CHECKPOINT_DIR,
        "pretrained_ViT": False, # Set True if you want to load ImageNet weights (requires download)
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

    # Transforms (Resize and Normalize)
    preprocess = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet Standard
            std=[0.229, 0.224, 0.225]), 
    ])

    print("Initializing Dataset...")
    # Check if image dir exists
    if not os.path.exists(IMAGE_DIR):
        print(f"WARNING: Image directory '{IMAGE_DIR}' does not exist.")
        print("Please create it and put your images (0.jpg, 1.jpg, ...) inside.")
    
    try:
        full_dataset = CustomJsonDataset(
            json_path=JSON_PATH, 
            image_dir=IMAGE_DIR, 
            window_size=args["window_size"], 
            transform=preprocess
        )
        
        # Split Train/Val
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args["bsize"], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args["bsize"], shuffle=False, num_workers=4)
        
        print(f"Dataset Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Continuing to model build to verify syntax, but training will fail.")
        train_loader = None
        val_loader = None

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = build_model(model_params)
    model = model.to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    criterion = torch.nn.MSELoss()
    tensorboard_writer = SummaryWriter(log_dir=os.path.join(args["checkpoint_path"], "logs"))

    # Training Loop
    if train_loader:
        best_val_loss = float('inf')
        
        for epoch in range(1, args["epoch"] + 1):
            print(f"\nStart Epoch {epoch}")
            
            # Train
            model.train()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args)
            
            # Val
            val_loss = val_epoch(model, val_loader, criterion, args)
            
            print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Save Best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args["checkpoint_path"], "best_model.pth"))
                print("Saved Best Model")
                
            # Save Checkpoint
            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(args["checkpoint_path"], f"checkpoint_e{epoch}.pth"))

    print("Done.")
