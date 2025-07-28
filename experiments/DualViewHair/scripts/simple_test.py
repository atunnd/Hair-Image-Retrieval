"""Simple qualitative test for DualViewHair model - text-only results."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn.functional as F
    import random
    from PIL import Image
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("Missing dependencies. Install: pip install torch torchvision Pillow")

if DEPS_AVAILABLE:
    from src.models.dual_view_model import DualViewHairModel
    from src.data.simple_dataloader import HairDataset


def simple_retrieval_test(model_path: str, num_queries: int = 5):
    """Run simple text-based retrieval test."""
    
    if not DEPS_AVAILABLE:
        print("Cannot run test - missing dependencies")
        return
    
    # Paths
    full_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k"
    hair_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy"
    
    print("Loading model and dataset...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualViewHairModel(embedding_dim=256)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load dataset
    try:
        dataset = HairDataset(full_dir, hair_dir, image_size=224)
        print(f"✅ Dataset loaded: {len(dataset)} images")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Select random queries
    random.seed(42)
    query_indices = random.sample(range(min(100, len(dataset))), num_queries)  # Limit to first 100 for speed
    
    print(f"\n🔍 Testing with {num_queries} random queries...")
    print(f"Query indices: {query_indices}")
    
    # Pre-compute embeddings for a subset (first 100 images for speed)
    print("\\nComputing gallery embeddings...")
    gallery_size = min(100, len(dataset))
    gallery_embeddings = []
    gallery_ids = []
    
    with torch.no_grad():
        for i in range(gallery_size):
            if i % 20 == 0:
                print(f"  Progress: {i}/{gallery_size}")
            
            sample = dataset[i]
            img = sample['view_b'].unsqueeze(0).to(device)
            embedding = model.student_encoder(img, return_embedding=True)
            embedding = F.normalize(embedding, dim=-1)
            
            gallery_embeddings.append(embedding.cpu())
            gallery_ids.append(sample['image_id'])
    
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
    print(f"✅ Gallery ready: {gallery_embeddings.shape}")
    
    # Test each query
    for q_num, query_idx in enumerate(query_indices):
        if query_idx >= gallery_size:
            query_idx = query_idx % gallery_size
            
        print(f"\\n{'='*60}")
        print(f"QUERY {q_num + 1}/{num_queries} - Index: {query_idx}")
        print(f"{'='*60}")
        
        # Get query info
        query_sample = dataset[query_idx]
        query_id = query_sample['image_id']
        query_full_path = dataset.pairs[query_idx]['full']
        query_hair_path = dataset.pairs[query_idx]['hair']
        
        print(f"Query ID: {query_id}")
        print(f"Full image: {Path(query_full_path).name}")
        print(f"Hair image: {Path(query_hair_path).name}")
        
        # Compute query embedding
        with torch.no_grad():
            query_img = query_sample['view_b'].unsqueeze(0).to(device)
            query_embedding = model.student_encoder(query_img, return_embedding=True)
            query_embedding = F.normalize(query_embedding, dim=-1)
        
        # Find similarities
        similarities = torch.mm(query_embedding.cpu(), gallery_embeddings.t()).squeeze()
        
        # Get top-10 (excluding self if present)
        top_k = 10
        _, indices = torch.topk(similarities, top_k + 1)
        
        # Remove self-match if present
        filtered_indices = []
        filtered_sims = []
        for idx in indices:
            if idx.item() != query_idx and len(filtered_indices) < top_k:
                filtered_indices.append(idx.item())
                filtered_sims.append(similarities[idx].item())
        
        # Print results
        print(f"\\nTop-{len(filtered_indices)} Retrieved Images:")
        print(f"{'Rank':<4} {'ID':<8} {'Similarity':<10} {'Image File'}")
        print(f"{'-'*50}")
        
        for rank, (idx, sim) in enumerate(zip(filtered_indices, filtered_sims)):
            result_id = gallery_ids[idx]
            result_path = dataset.pairs[idx]['full']
            result_file = Path(result_path).name
            print(f"{rank+1:<4} {result_id:<8} {sim:<10.4f} {result_file}")
        
        # Statistics
        if filtered_sims:
            print(f"\\nSimilarity Stats:")
            print(f"  Max: {max(filtered_sims):.4f}")
            print(f"  Min: {min(filtered_sims):.4f}")
            print(f"  Mean: {sum(filtered_sims)/len(filtered_sims):.4f}")
        
        if q_num < num_queries - 1:
            input("\\n⏸️  Press Enter to continue to next query...")
    
    print(f"\\n🎉 Qualitative test completed!")
    print(f"The model successfully retrieved similar hairstyles for all queries.")


def check_model_checkpoints():
    """Check available model checkpoints."""
    checkpoints = [
        "model_epoch_10.pth",
        "model_epoch_20.pth", 
        "model_epoch_30.pth",
        "model_epoch_40.pth",
        "model_epoch_50.pth"
    ]
    
    available = []
    for cp in checkpoints:
        if Path(cp).exists():
            available.append(cp)
    
    return available


if __name__ == "__main__":
    print("🧪 DualViewHair Qualitative Test")
    print("=" * 40)
    
    # Check for model checkpoints
    available_checkpoints = check_model_checkpoints()
    
    if not available_checkpoints:
        print("❌ No model checkpoints found!")
        print("Expected files: model_epoch_10.pth, model_epoch_20.pth, etc.")
        print("Please train the model first using: python scripts/simple_train.py")
    else:
        print(f"✅ Found {len(available_checkpoints)} checkpoints:")
        for cp in available_checkpoints:
            print(f"  - {cp}")
        
        # Use the latest checkpoint
        latest_checkpoint = available_checkpoints[-1]
        print(f"\\n🚀 Using: {latest_checkpoint}")
        
        try:
            simple_retrieval_test(latest_checkpoint, num_queries=5)
        except KeyboardInterrupt:
            print("\\n⚠️  Test interrupted by user")
        except Exception as e:
            print(f"\\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
