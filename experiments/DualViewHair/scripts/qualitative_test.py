"""Qualitative test for DualViewHair model - retrieve top-10 similar hairstyles."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.models.dual_view_model import DualViewHairModel
from src.data.simple_dataloader import HairDataset


class HairRetrieval:
    """Hairstyle retrieval system using trained DualViewHair model."""
    
    def __init__(self, model_path: str, full_dir: str, hair_dir: str, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.full_dir = Path(full_dir)
        self.hair_dir = Path(hair_dir)
        
        # Load model
        self.model = DualViewHairModel(embedding_dim=256)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # Create dataset for gallery
        self.dataset = HairDataset(full_dir, hair_dir, image_size=224)
        print(f"Gallery size: {len(self.dataset)} images")
        
        # Pre-compute all embeddings
        self.gallery_embeddings = None
        self.gallery_ids = None
        self._build_gallery()
    
    def _build_gallery(self):
        """Pre-compute embeddings for all gallery images."""
        print("Building gallery embeddings...")
        
        embeddings = []
        ids = []
        
        with torch.no_grad():
            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                full_img = sample['view_b'].unsqueeze(0).to(self.device)  # Use full image
                
                # Get embedding using student encoder
                embedding = self.model.student_encoder(full_img, return_embedding=True)
                embedding = F.normalize(embedding, dim=-1)
                
                embeddings.append(embedding.cpu())
                ids.append(sample['image_id'])
        
        self.gallery_embeddings = torch.cat(embeddings, dim=0)  # [N, embed_dim]
        self.gallery_ids = ids
        print(f"Gallery built: {self.gallery_embeddings.shape}")
    
    def retrieve_similar(self, query_idx: int, top_k: int = 10):
        """Retrieve top-k similar hairstyles for a query image."""
        
        # Get query embedding
        query_sample = self.dataset[query_idx]
        query_img = query_sample['view_b'].unsqueeze(0).to(self.device)
        query_id = query_sample['image_id']
        
        with torch.no_grad():
            query_embedding = self.model.student_encoder(query_img, return_embedding=True)
            query_embedding = F.normalize(query_embedding, dim=-1)
        
        # Compute similarities
        similarities = torch.mm(query_embedding.cpu(), self.gallery_embeddings.t()).squeeze()
        
        # Get top-k results (excluding self)
        _, indices = torch.topk(similarities, top_k + 1)  # +1 to exclude self
        indices = indices[1:]  # Remove self-match
        top_similarities = similarities[indices]
        
        # Get result info
        results = []
        for idx, sim in zip(indices, top_similarities):
            results.append({
                'gallery_idx': idx.item(),
                'image_id': self.gallery_ids[idx],
                'similarity': sim.item(),
                'full_path': self.dataset.pairs[idx]['full'],
                'hair_path': self.dataset.pairs[idx]['hair']
            })
        
        return {
            'query_idx': query_idx,
            'query_id': query_id,
            'query_full_path': self.dataset.pairs[query_idx]['full'],
            'query_hair_path': self.dataset.pairs[query_idx]['hair'],
            'results': results
        }
    
    def visualize_retrieval(self, retrieval_result, save_path=None):
        """Visualize query and top-10 retrieval results."""
        
        query_info = retrieval_result
        results = query_info['results']
        
        # Create figure
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        fig.suptitle(f'Hairstyle Retrieval - Query ID: {query_info["query_id"]}', fontsize=16)
        
        # Load and display query (both full and hair)
        query_full = Image.open(query_info['query_full_path']).convert('RGB')
        query_hair = Image.open(query_info['query_hair_path']).convert('RGB')
        
        axes[0, 0].imshow(query_full)
        axes[0, 0].set_title('Query\n(Full Image)', fontweight='bold', color='red')
        axes[0, 0].axis('off')
        axes[0, 0].add_patch(patches.Rectangle((0, 0), query_full.width-1, query_full.height-1, 
                                              linewidth=3, edgecolor='red', facecolor='none'))
        
        axes[1, 0].imshow(query_hair)
        axes[1, 0].set_title('Query\n(Hair Only)', fontweight='bold', color='red')
        axes[1, 0].axis('off')
        axes[1, 0].add_patch(patches.Rectangle((0, 0), query_hair.width-1, query_hair.height-1, 
                                              linewidth=3, edgecolor='red', facecolor='none'))
        
        # Display top-5 results
        for i, result in enumerate(results[:5]):
            col = i + 1
            
            # Load images
            result_full = Image.open(result['full_path']).convert('RGB')
            result_hair = Image.open(result['hair_path']).convert('RGB')
            
            # Display full image
            axes[0, col].imshow(result_full)
            axes[0, col].set_title(f'Rank {i+1}\nSim: {result["similarity"]:.3f}', fontsize=10)
            axes[0, col].axis('off')
            
            # Display hair image
            axes[1, col].imshow(result_hair)
            axes[1, col].set_title(f'ID: {result["image_id"]}', fontsize=8)
            axes[1, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def print_retrieval_stats(self, retrieval_result):
        """Print detailed retrieval statistics."""
        results = retrieval_result['results']
        
        print(f"\n{'='*60}")
        print(f"RETRIEVAL RESULTS - Query ID: {retrieval_result['query_id']}")
        print(f"{'='*60}")
        
        similarities = [r['similarity'] for r in results]
        print(f"Similarity Range: {min(similarities):.4f} - {max(similarities):.4f}")
        print(f"Mean Similarity: {np.mean(similarities):.4f}")
        print(f"Std Similarity: {np.std(similarities):.4f}")
        
        print(f"\nTop-10 Results:")
        print(f"{'Rank':<4} {'ID':<8} {'Similarity':<10} {'Image Path'}")
        print(f"{'-'*60}")
        
        for i, result in enumerate(results):
            path_short = Path(result['full_path']).name
            print(f"{i+1:<4} {result['image_id']:<8} {result['similarity']:<10.4f} {path_short}")


def run_qualitative_test(model_path: str, num_queries: int = 5):
    """Run qualitative test with random queries."""
    
    # Paths
    full_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k"
    hair_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy"
    
    # Initialize retrieval system
    print("Initializing HairRetrieval system...")
    retrieval = HairRetrieval(model_path, full_dir, hair_dir)
    
    # Select random query images
    random.seed(42)  # For reproducible results
    query_indices = random.sample(range(len(retrieval.dataset)), num_queries)
    
    print(f"\nRunning qualitative test with {num_queries} random queries...")
    print(f"Selected query indices: {query_indices}")
    
    # Process each query
    for i, query_idx in enumerate(query_indices):
        print(f"\n{'='*80}")
        print(f"QUERY {i+1}/{num_queries} - Index: {query_idx}")
        print(f"{'='*80}")
        
        # Retrieve similar images
        result = retrieval.retrieve_similar(query_idx, top_k=10)
        
        # Print statistics
        retrieval.print_retrieval_stats(result)
        
        # Create visualization
        output_dir = Path("retrieval_results")
        output_dir.mkdir(exist_ok=True)
        
        save_path = output_dir / f"query_{i+1}_id_{result['query_id']}.png"
        retrieval.visualize_retrieval(result, save_path)
        
        # Wait for user input to continue
        if i < num_queries - 1:
            input("\nPress Enter to continue to next query...")
    
    print(f"\n{'='*80}")
    print("QUALITATIVE TEST COMPLETED!")
    print(f"Results saved in: ./retrieval_results/")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Test with latest checkpoint
    model_checkpoints = [
        "model_epoch_10.pth",
        "model_epoch_20.pth", 
        "model_epoch_30.pth",
        "model_epoch_40.pth",
        "model_epoch_50.pth"
    ]
    
    # Use the latest available checkpoint
    model_path = None
    for checkpoint in reversed(model_checkpoints):
        if Path(checkpoint).exists():
            model_path = checkpoint
            break
    
    if model_path is None:
        print("No model checkpoint found! Please train the model first.")
        print("Expected files: model_epoch_10.pth, model_epoch_20.pth, etc.")
    else:
        print(f"Using checkpoint: {model_path}")
        run_qualitative_test(model_path, num_queries=5)
