"""
Simple NT-Xent training script for DualViewHair.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from tqdm import tqdm

from src.models.dual_view_model import DualViewHairModel
from src.data.simple_dataloader import create_dataloader
from src.losses.ntxent_loss import NTXentLoss, AsymmetricNTXentLoss
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Fix PIL image loading issues


def train_epoch(model, loader, optimizer, criterion, device):
    """Single training epoch with NT-Xent loss."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        view_a = batch['view_a'].to(device)  # Hair regions
        view_b = batch['view_b'].to(device)  # Full images
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(view_a, view_b)
        student_proj = outputs['student_projection']
        teacher_proj = outputs['teacher_projection']
        
        # NT-Xent loss
        loss = criterion(student_proj, teacher_proj)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    """Main training function."""
    
    # Paths
    full_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k"
    hair_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy"
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_loader = create_dataloader(
        full_dir, hair_dir, 
        batch_size=64,  # Reduced for NT-Xent (2x memory usage)
        num_workers=4
    )
    
    # Model
    model = DualViewHairModel(
        embedding_dim=256,
        projection_dim=128
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss function - choose one:
    # criterion = NTXentLoss(temperature=0.07)  # Symmetric
    criterion = AsymmetricNTXentLoss(  # Asymmetric (recommended)
        temperature=0.07,
        student_weight=1.0,
        teacher_weight=0.5
    )
    
    print(f"Training with {criterion.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Dataset size: {len(train_loader.dataset)}")
    
    # Training loop
    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1:2d}: Loss = {train_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"model_ntxent_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'criterion': criterion.__class__.__name__
            }, checkpoint_path)
            print(f"  Saved: {checkpoint_path}")
    
    print("NT-Xent training complete!")


if __name__ == "__main__":
    main()
