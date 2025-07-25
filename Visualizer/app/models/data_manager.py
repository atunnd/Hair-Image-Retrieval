"""
Enhanced data loader with support for multiple datasets and cleaner organization.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Union
from fastapi import HTTPException
import logging

from app.config.datasets import get_dataset_config, get_result_file_path, list_available_datasets, list_available_models

logger = logging.getLogger(__name__)

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in file: {path}")

def load_benchmark(dataset_key: str) -> Dict[str, List[str]]:
    """Load benchmark data for a specific dataset."""
    dataset_config = get_dataset_config(dataset_key)
    benchmark_file = dataset_config["benchmark_file"]
    
    if not benchmark_file.exists():
        logger.error(f"Benchmark file not found: {benchmark_file}")
        return {}
    
    data = load_json(benchmark_file)
    result = {}
    
    for entry in data:
        query_image = entry["query_image"]
        
        # Handle different query image formats based on dataset
        if dataset_key == "korean":
            # Korean format: "JS596196_query.jpg" -> "JS596196"
            if "_query.jpg" in query_image:
                key = query_image.replace("_query.jpg", "")
            else:
                key = query_image.replace(".jpg", "")
        else:
            # Regular hairstyle format: "3708.jpg" -> "3708"
            key = query_image.replace(".jpg", "")
        
        result[key] = entry["ground_truth"]
    
    logger.debug(f"Loaded benchmark for {dataset_key}: {len(result)} queries")
    return result

def load_model_results(dataset_key: str, model_key: str) -> Dict[str, List[str]]:
    """Load model results for a specific dataset and model."""
    try:
        result_file = get_result_file_path(dataset_key, model_key)
        
        if not result_file.exists():
            logger.error(f"Result file not found: {result_file}")
            return {}
        
        raw_data = load_json(result_file)
        
        if not raw_data or len(raw_data) == 0:
            return {}
        
        # Handle different result formats
        first_entry = raw_data[0]
        
        if "query" in first_entry:
            # Original format: {"query": "3708_hair.png", "top100": [...]}
            results = {entry["query"]: entry["top100"] for entry in raw_data}
        elif "query_id" in first_entry:
            # Korean format: {"query_id": "JS596196_query", "top100": [...]}
            results = {entry["query_id"]: entry["top100"] for entry in raw_data}
        else:
            logger.error(f"Unknown result format in {result_file}")
            return {}
        
        logger.debug(f"Loaded {model_key} results for {dataset_key}: {len(results)} queries")
        return results
        
    except Exception as e:
        logger.error(f"Error loading model results for {dataset_key}/{model_key}: {e}")
        return {}

def load_all_datasets() -> Dict[str, Dict[str, List[str]]]:
    """Load all available datasets."""
    datasets = {}
    for dataset_key in list_available_datasets():
        datasets[dataset_key] = load_benchmark(dataset_key)
    return datasets

def load_all_model_results() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Load all model results for all datasets."""
    all_results = {}
    
    for dataset_key in list_available_datasets():
        all_results[dataset_key] = {}
        for model_key in list_available_models():
            all_results[dataset_key][model_key] = load_model_results(dataset_key, model_key)
    
    return all_results

class DataManager:
    """Centralized data management class."""
    
    def __init__(self):
        self.datasets = {}
        self.model_results = {}
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all datasets and model results."""
        try:
            self.datasets = load_all_datasets()
            self.model_results = load_all_model_results()
            logger.info("Successfully loaded all data")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self._initialize_empty_data()
    
    def _initialize_empty_data(self):
        """Initialize empty data structures."""
        self.datasets = {dataset: {} for dataset in list_available_datasets()}
        self.model_results = {
            dataset: {model: {} for model in list_available_models()}
            for dataset in list_available_datasets()
        }
    
    def get_benchmark(self, dataset_key: str) -> Dict[str, List[str]]:
        """Get benchmark data for a dataset."""
        return self.datasets.get(dataset_key, {})
    
    def get_model_results(self, dataset_key: str, model_key: str) -> Dict[str, List[str]]:
        """Get model results for a specific dataset and model."""
        return self.model_results.get(dataset_key, {}).get(model_key, {})
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return list(self.datasets.keys())
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list_available_models()
    
    def reload_data(self):
        """Reload all data."""
        self._load_all_data()
