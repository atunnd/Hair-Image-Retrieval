"""
Dataset configuration for the Hairstyle Retrieval Visualizer.
This file centralizes all dataset and model configuration.
"""

from pathlib import Path
from typing import Dict, Any, List

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

def get_result_file_path(dataset_key: str, model_key: str, version: str = "10k") -> Path:
    """Get the path to a model's results file for a dataset."""
    dataset_config = get_dataset_config(dataset_key)
    model_config = get_model_config(model_key)
    
    if dataset_key == "hairstyle":
        filename = f"{model_key}_{version}.json"
    elif dataset_key == "korean":
        filename = f"{model_key}_k_hairstyle_{version}.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    
    return dataset_config["results_dir"] / model_key / filename

def get_available_model_versions(dataset_key: str, model_key: str) -> List[str]:
    """Get available versions for a specific model and dataset."""
    dataset_config = get_dataset_config(dataset_key)
    model_dir = dataset_config["results_dir"] / model_key
    
    if not model_dir.exists():
        return []
    
    versions = []
    for file_path in model_dir.glob("*.json"):
        filename = file_path.stem
        if dataset_key == "hairstyle":
            # Extract version from filename like "dino_10k.json" -> "10k"
            if filename.startswith(f"{model_key}_"):
                version = filename.replace(f"{model_key}_", "")
                versions.append(version)
        elif dataset_key == "korean":
            # Extract version from filename like "dino_k_hairstyle_10k.json" -> "10k"
            if filename.startswith(f"{model_key}_k_hairstyle_"):
                version = filename.replace(f"{model_key}_k_hairstyle_", "")
                versions.append(version)
    
    return sorted(versions)

def get_all_available_versions() -> Dict[str, Dict[str, List[str]]]:
    """Get all available versions for all models and datasets."""
    all_versions = {}
    for dataset_key in DATASETS.keys():
        all_versions[dataset_key] = {}
        for model_key in MODELS.keys():
            all_versions[dataset_key][model_key] = get_available_model_versions(dataset_key, model_key)
    return all_versions

def list_available_datasets() -> list:
    """List all available datasets."""
    return list(DATASETS.keys())

def list_available_models() -> list:
    """List all available models."""
    return list(MODELS.keys())
