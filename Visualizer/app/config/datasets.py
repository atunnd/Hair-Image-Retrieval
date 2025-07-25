"""
Dataset configuration for the Hairstyle Retrieval Visualizer.
This file centralizes all dataset and model configuration.
"""

from pathlib import Path
from typing import Dict, Any

# Base data directory
BASE_DATA_DIR = Path("data")

# Dataset configurations
DATASETS = {
    "hairstyle": {
        "name": "Hairstyle Retrieval",
        "benchmark_file": BASE_DATA_DIR / "hairstyle_retrieval_benchmark.json",
        "results_dir": BASE_DATA_DIR / "hairstyle_retrieval",
        "image_type": "hair_face_split",  # Uses separate hair and face images
        "image_mounts": {
            "hair": "/hair_images",
            "face": "/face_images"
        },
        "image_paths": {
            "hair": "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/HairLearning/data/train/HairImages/",
            "face": "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/HairstyleRetrieval/data/FrontalFaceNoBg/FrontalFaceNoBg"
        }
    },
    "korean": {
        "name": "Korean Hairstyle Retrieval",
        "benchmark_file": BASE_DATA_DIR / "korean_hairstyle_retrieval_benchmark.json",
        "results_dir": BASE_DATA_DIR / "k-hairstyle",
        "image_type": "single",  # Uses single image files
        "image_mounts": {
            "main": "/korean_images"
        },
        "image_paths": {
            "main": "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/data/korean_hairstyle_benchmark/images/"
        }
    }
}

# Model configurations
MODELS = {
    "dino": {
        "name": "DINO",
        "description": "Vision Transformer with DINO self-supervised learning"
    },
    "simmim": {
        "name": "SimMIM", 
        "description": "Masked Image Modeling for self-supervised learning"
    },
    "mae": {
        "name": "MAE",
        "description": "Masked Autoencoders for self-supervised learning"
    },
    "siamim": {
        "name": "SiamIM",
        "description": "Siamese Image Modeling"
    },
    "simclr": {
        "name": "SimCLR",
        "description": "Contrastive Learning of Visual Representations"
    }
}

def get_dataset_config(dataset_key: str) -> Dict[str, Any]:
    """Get configuration for a specific dataset."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    return DATASETS[dataset_key]

def get_model_config(model_key: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    return MODELS[model_key]

def get_result_file_path(dataset_key: str, model_key: str) -> Path:
    """Get the path to a model's results file for a dataset."""
    dataset_config = get_dataset_config(dataset_key)
    model_config = get_model_config(model_key)
    
    if dataset_key == "hairstyle":
        filename = f"{model_key}_10k.json"
    elif dataset_key == "korean":
        filename = f"{model_key}_k_hairstyle_10k.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    
    return dataset_config["results_dir"] / model_key / filename

def list_available_datasets() -> list:
    """List all available datasets."""
    return list(DATASETS.keys())

def list_available_models() -> list:
    """List all available models."""
    return list(MODELS.keys())
