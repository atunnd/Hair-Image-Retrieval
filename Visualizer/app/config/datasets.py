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
def get_dynamic_models() -> Dict[str, Dict[str, str]]:
    """Dynamically generate model configurations based on available files."""
    models = {}
    all_versions = get_all_available_versions()
    
    base_models = {
        "dino": {
            "name": "DINO",
            "description": "Self-supervised Vision Transformer"
        },
        "simmim": {
            "name": "SimMIM", 
            "description": "Masked Image Modeling"
        },
        "mae": {
            "name": "MAE",
            "description": "Masked Autoencoder"
        },
        "siamim": {
            "name": "SiamIM",
            "description": "Siamese Masked Image Modeling"
        },
        "simclr": {
            "name": "SimCLR",
            "description": "Contrastive Learning"
        }
    }
    
    # Create model entries for each version found
    for dataset_key in all_versions:
        for model_key, versions in all_versions[dataset_key].items():
            if model_key in base_models:
                base_config = base_models[model_key]
                for version in versions:
                    model_version_key = f"{model_key}_{version}"
                    models[model_version_key] = {
                        "name": f"{base_config['name']} ({version})",
                        "description": base_config["description"],
                        "base_model": model_key,
                        "version": version
                    }
    
    return models

# For backward compatibility, we'll also keep a static version
MODELS = {
    "dino": {
        "name": "DINO",
        "description": "Self-supervised Vision Transformer"
    },
    "simmim": {
        "name": "SimMIM", 
        "description": "Masked Image Modeling"
    },
    "mae": {
        "name": "MAE",
        "description": "Masked Autoencoder"
    },
    "siamim": {
        "name": "SiamIM",
        "description": "Siamese Masked Image Modeling"
    },
    "simclr": {
        "name": "SimCLR",
        "description": "Contrastive Learning"
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
    """List all available base models."""
    return list(MODELS.keys())

def list_all_model_versions() -> List[str]:
    """List all model versions (base_model_version format)."""
    return list(get_dynamic_models().keys())

def parse_model_version_key(model_version_key: str) -> tuple[str, str]:
    """Parse a model version key like 'siamim_60k' into ('siamim', '60k')."""
    dynamic_models = get_dynamic_models()
    if model_version_key in dynamic_models:
        config = dynamic_models[model_version_key]
        return config['base_model'], config['version']
    
    # Fallback for direct base model names
    if model_version_key in MODELS:
        return model_version_key, "10k"
    
    # Try to parse manually
    parts = model_version_key.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    
    return model_version_key, "10k"
