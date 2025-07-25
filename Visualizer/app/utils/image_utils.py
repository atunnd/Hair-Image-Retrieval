"""
Image handling utilities for different dataset formats.
"""

from typing import Dict, List, Tuple
from app.config.datasets import get_dataset_config

class ImagePathResolver:
    """Handles image path resolution for different dataset formats."""
    
    @staticmethod
    def get_query_image_path(dataset_key: str, query_id: str, view_mode: str = "full") -> str:
        """Get the path for a query image."""
        dataset_config = get_dataset_config(dataset_key)
        
        if dataset_config["image_type"] == "hair_face_split":
            # Regular hairstyle dataset
            if view_mode == "hair":
                return f"{dataset_config['image_mounts']['hair']}/{query_id}_hair.png"
            else:
                return f"{dataset_config['image_mounts']['face']}/{query_id}.jpg"
        
        elif dataset_config["image_type"] == "single":
            # Korean hairstyle dataset
            return f"{dataset_config['image_mounts']['main']}/{query_id}_query.jpg"
        
        else:
            raise ValueError(f"Unknown image type: {dataset_config['image_type']}")
    
    @staticmethod
    def get_result_image_paths(dataset_key: str, image_filename: str) -> Dict[str, str]:
        """Get hair and face image paths for a result image."""
        dataset_config = get_dataset_config(dataset_key)
        
        if dataset_config["image_type"] == "hair_face_split":
            # Regular hairstyle dataset
            hair_path = f"{dataset_config['image_mounts']['hair']}/{image_filename}"
            face_path = f"{dataset_config['image_mounts']['face']}/{image_filename.replace('_hair.png', '.jpg')}"
            return {"hair": hair_path, "face": face_path}
        
        elif dataset_config["image_type"] == "single":
            # Korean hairstyle dataset - same image for both
            image_path = f"{dataset_config['image_mounts']['main']}/{image_filename}"
            return {"hair": image_path, "face": image_path}
        
        else:
            raise ValueError(f"Unknown image type: {dataset_config['image_type']}")
    
    @staticmethod
    def get_ground_truth_paths(dataset_key: str, ground_truth_files: List[str]) -> List[str]:
        """Get paths for ground truth images."""
        dataset_config = get_dataset_config(dataset_key)
        
        if dataset_config["image_type"] == "hair_face_split":
            # Regular hairstyle dataset - use face images for ground truth
            return [f"{dataset_config['image_mounts']['face']}/{filename}" for filename in ground_truth_files]
        
        elif dataset_config["image_type"] == "single":
            # Korean hairstyle dataset
            return [f"{dataset_config['image_mounts']['main']}/{filename}" for filename in ground_truth_files]
        
        else:
            raise ValueError(f"Unknown image type: {dataset_config['image_type']}")

class ResultProcessor:
    """Processes model results and computes hits/misses."""
    
    @staticmethod
    def get_query_key(dataset_key: str, query_id: str) -> str:
        """Get the query key used in model results."""
        if dataset_key == "korean":
            # Korean dataset uses "JS596196_query" format
            return f"{query_id}_query" if not query_id.endswith("_query") else query_id
        else:
            # Regular dataset uses "3708_hair.png" format
            return f"{query_id}_hair.png"
    
    @staticmethod
    def compute_hits_and_misses(dataset_key: str, model_results: List[str], ground_truth: List[str]) -> Tuple[List[str], List[str]]:
        """Compute hits and misses for model results."""
        if dataset_key == "korean":
            # Korean dataset: direct comparison
            hits = [img for img in model_results if img in ground_truth]
            misses = [img for img in model_results if img not in hits]
        else:
            # Regular dataset: convert hair images to face images for comparison
            hits = [img for img in model_results if img.replace('_hair.png', '.jpg') in ground_truth]
            misses = [img for img in model_results if img not in hits]
        
        return hits, misses
