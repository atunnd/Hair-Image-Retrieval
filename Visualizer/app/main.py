from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.models.retrieval_api import router as retrieval_router
from app.models.data_manager import DataManager
from app.utils.image_utils import ImagePathResolver, ResultProcessor
from app.config.datasets import DATASETS, MODELS, list_available_datasets, list_available_models
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hairstyle Retrieval Visualizer", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files based on dataset configurations
for dataset_key, dataset_config in DATASETS.items():
    for mount_key, mount_path in dataset_config["image_mounts"].items():
        static_dir = dataset_config["image_paths"][mount_key]
        mount_name = f"{dataset_key}_{mount_key}" if mount_key != "main" else dataset_key
        app.mount(mount_path, StaticFiles(directory=static_dir), name=mount_name)
        logger.info(f"Mounted {mount_path} -> {static_dir}")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Include API routes
app.include_router(retrieval_router, prefix="/api")

# Initialize data manager
data_manager = DataManager()

@app.get("/", response_class=HTMLResponse)
@app.post("/", response_class=HTMLResponse)
async def index(
    request: Request,
    dataset: str = Form("hairstyle"),
    query_id: str = Form(None),
    query_index: int = Form(0),
    models: list = Form(["dino", "simmim"]),
    show_only_correct: bool = Form(False),
    view_mode: str = Form("full")
):
    """Main interface for the hairstyle retrieval visualizer."""
    
    # Get dataset and validate
    available_datasets = data_manager.get_available_datasets()
    if dataset not in available_datasets:
        dataset = available_datasets[0] if available_datasets else "hairstyle"
    
    # Get benchmark data
    current_benchmark = data_manager.get_benchmark(dataset)
    queries = list(current_benchmark.keys())
    
    # Handle query selection
    if query_id and query_id in queries:
        selected_query = query_id
        query_index = queries.index(query_id)
    else:
        query_index = max(0, min(query_index, len(queries) - 1))
        selected_query = queries[query_index] if queries else ""
    
    logger.debug(f"Dataset: {dataset}, Query: {selected_query}, Models: {models}")
    
    # Limit to first two models for display
    display_models = models[:2]
    available_models = data_manager.get_available_models()
    
    # Process results for each model
    results = {}
    for model in display_models:
        if model not in available_models:
            continue
            
        # Get model results
        model_results_raw = data_manager.get_model_results(dataset, model)
        query_key = ResultProcessor.get_query_key(dataset, selected_query)
        model_result = model_results_raw.get(query_key, [])
        
        # Get ground truth
        ground_truth = current_benchmark.get(selected_query, [])
        
        # Compute hits and misses
        hits, misses = ResultProcessor.compute_hits_and_misses(dataset, model_result, ground_truth)
        
        # Filter results if requested
        images_to_show = hits if show_only_correct else model_result
        
        # Generate image paths
        result_images = []
        for img in images_to_show:
            img_paths = ImagePathResolver.get_result_image_paths(dataset, img)
            result_images.append(img_paths)
        
        hit_images = []
        for img in hits:
            img_paths = ImagePathResolver.get_result_image_paths(dataset, img)
            hit_images.append(img_paths)
        
        miss_images = []
        for img in misses:
            img_paths = ImagePathResolver.get_result_image_paths(dataset, img)
            miss_images.append(img_paths)
        
        # Store results
        results[model] = {
            "model_name": MODELS[model]["name"],
            "model_description": MODELS[model]["description"],
            "top100": result_images,
            "ground_truth": ImagePathResolver.get_ground_truth_paths(dataset, ground_truth),
            "hits": hit_images,
            "misses": miss_images,
            "stats": {
                "total_results": len(model_result),
                "hits_count": len(hits),
                "misses_count": len(misses),
                "accuracy": len(hits) / len(model_result) if model_result else 0
            }
        }
        
        logger.debug(f"Model {model} - Hits: {len(hits)}, Misses: {len(misses)}")
    
    # Get query image path
    query_image_path = ImagePathResolver.get_query_image_path(dataset, selected_query, view_mode)
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "datasets": DATASETS,
            "models": MODELS,
            "selected_dataset": dataset,
            "queries": queries,
            "query_index": query_index,
            "selected_query": selected_query,
            "selected_models": models,
            "display_models": display_models,
            "show_only_correct": show_only_correct,
            "view_mode": view_mode,
            "query_image_path": query_image_path,
            "results": results,
            "available_datasets": available_datasets,
            "available_models": available_models
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "datasets_loaded": len(data_manager.get_available_datasets()),
        "models_available": len(data_manager.get_available_models())
    }

@app.post("/reload")
async def reload_data():
    """Reload all data (useful for development)."""
    data_manager.reload_data()
    return {"status": "reloaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)