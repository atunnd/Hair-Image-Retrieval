from fastapi import APIRouter, HTTPException
from app.models.data_loader import load_model_results, load_benchmark
from app.schemas.retrieval import QueryResult, AvailableModelsResponse, AvailableQueriesResponse
import logging

router = APIRouter()

# Set up logging
logger = logging.getLogger(__name__)

# Load once at startup
benchmark_path = "data/hairstyle_retrieval_benchmark.json"
model_files = {
    "dino": "data/dino_top100_results.json",
    "simmim": "data/simmim_top100_results.json",
    "mae": "data/mae_top100_results.json",
    "siamim": "data/siamim_top100_results.json",
    "simclr": "data/simclr_top100_results.json"
}

try:
    benchmark = load_benchmark(benchmark_path)
    models_data = {name: load_model_results(path) for name, path in model_files.items()}
    logger.debug(f"API - Benchmark keys: {list(benchmark.keys())}")
    logger.debug(f"API - Models data keys: { {name: list(data.keys()) for name, data in models_data.items()} }")
except Exception as e:
    logger.error(f"API - Error loading data: {e}")
    benchmark = {}
    models_data = {"dino": {}, "simmim": {}, "mae": {}, "siamim": {}, "simclr": {}}

@router.get("/models", response_model=AvailableModelsResponse)
def list_models():
    models = list(models_data.keys())
    logger.debug(f"Returning models: {models}")
    return {"models": models}

@router.get("/queries", response_model=AvailableQueriesResponse)
def list_queries():
    queries = list(benchmark.keys())
    logger.debug(f"Returning queries: {queries}")
    return {"queries": queries}

@router.get("/result", response_model=QueryResult)
def get_query_result(model: str, query_id: str):
    logger.debug(f"Fetching result for model: {model}, query_id: {query_id}")
    if model not in models_data:
        logger.error(f"Model not found: {model}")
        raise HTTPException(status_code=404, detail="Model not found")
    
    query_key = f"{query_id}_hair.png"
    model_result = models_data[model].get(query_key)
    ground_truth = benchmark.get(query_id, [])

    if model_result is None:
        logger.error(f"Query not found: {query_key}")
        raise HTTPException(status_code=404, detail="Query not found in model")

    hits = [img for img in model_result if img.replace('_hair.png', '.jpg') in ground_truth]
    misses = [img for img in model_result if img not in hits]

    logger.debug(f"Ground truth: {ground_truth}")
    logger.debug(f"Top100 (first 5): {model_result[:5]}")
    logger.debug(f"Hits: {hits}")

    return {
        "model": model,
        "query_id": query_id,
        "query_image": f"/hair_images/{query_id}_hair.png",
        "query_image_face": f"/face_images/{query_id}.jpg",
        "top100": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in model_result],
        "ground_truth": [f"/face_images/{img}" for img in ground_truth],
        "hits": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in hits],
        "misses": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in misses]
    }