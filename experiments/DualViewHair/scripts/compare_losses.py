"""
Comparison script to test different loss functions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from tqdm import tqdm

from src.models.dual_view_model import DualViewHairModel, ContrastiveLoss
from src.data.simple_dataloader import create_dataloader
from src.losses.ntxent_loss import NTXentLoss, AsymmetricNTXentLoss


def compare_losses():
    """Compare different loss functions on a small batch."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    embed_dim = 128
    
    # Create dummy data
    student_proj = torch.randn(batch_size, embed_dim).to(device)
    teacher_proj = torch.randn(batch_size, embed_dim).to(device)
    
    # Test different losses
    losses = {
        'InfoNCE (Original)': ContrastiveLoss(temperature=0.07),
        'NT-Xent (Symmetric)': NTXentLoss(temperature=0.07),
        'NT-Xent (Asymmetric)': AsymmetricNTXentLoss(temperature=0.07, student_weight=1.0, teacher_weight=0.5)
    }
    
    print("Loss Comparison on Random Batch:")
    print("-" * 40)
    
    for name, criterion in losses.items():
        with torch.no_grad():
            loss_value = criterion(student_proj, teacher_proj)
            print(f"{name:20s}: {loss_value.item():.4f}")
    
    print("\nMemory Usage Comparison:")
    print("-" * 40)
    
    # Memory usage comparison
    for name, criterion in losses.items():
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Simulate forward + backward
        student_proj_copy = student_proj.clone().requires_grad_(True)
        teacher_proj_copy = teacher_proj.clone().requires_grad_(True)
        
        loss = criterion(student_proj_copy, teacher_proj_copy)
        loss.backward()
        
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"{name:20s}: {memory_mb:.1f} MB")
            torch.cuda.reset_peak_memory_stats()
        else:
            print(f"{name:20s}: CPU mode")


def quick_training_test():
    """Quick training test with few epochs."""
    
    # Paths (adjust as needed)
    full_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k"
    hair_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small test setup
    train_loader = create_dataloader(
        full_dir, hair_dir,
        batch_size=16,  # Small batch for testing
        num_workers=2
    )
    
    # Test each loss function
    loss_configs = [
        ('InfoNCE', ContrastiveLoss(temperature=0.07)),
        ('NT-Xent', NTXentLoss(temperature=0.07)),
        ('Asymmetric NT-Xent', AsymmetricNTXentLoss(temperature=0.07, student_weight=1.0, teacher_weight=0.5))
    ]
    
    for loss_name, criterion in loss_configs:
        print(f"\nTesting {loss_name}...")
        print("-" * 30)
        
        # Fresh model for each test
        model = DualViewHairModel(embedding_dim=256, projection_dim=128).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train for 3 epochs
        for epoch in range(3):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                if batch_count >= 10:  # Only 10 batches per epoch for testing
                    break
                    
                view_a = batch['view_a'].to(device)
                view_b = batch['view_b'].to(device)
                
                optimizer.zero_grad()
                outputs = model(view_a, view_b)
                
                loss = criterion(outputs['student_projection'], outputs['teacher_projection'])
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            print(f"  Epoch {epoch+1}: {avg_loss:.4f}")


if __name__ == "__main__":
    print("DualViewHair Loss Comparison")
    print("=" * 50)
    
    # Run comparisons
    compare_losses()
    
    # Quick training test (uncomment to run)
    # quick_training_test()
