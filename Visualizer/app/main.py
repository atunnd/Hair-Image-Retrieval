from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.models.retrieval_api import router as retrieval_router
from app.models.data_loader import load_benchmark, load_model_results
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware (optional, for future JS if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/hair_images", StaticFiles(directory="/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/HairLearning/data/train/HairImages/"), name="hair_images")
app.mount("/face_images", StaticFiles(directory="/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/HairstyleRetrieval/data/FrontalFaceNoBg/FrontalFaceNoBg"), name="face_images")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Include API routes
app.include_router(retrieval_router, prefix="/api")

# Load data at startup
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
    logger.debug(f"Benchmark keys: {list(benchmark.keys())}")
    logger.debug(f"Models data keys: { {name: list(data.keys()) for name, data in models_data.items()} }")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    benchmark = {}
    models_data = {"dino": {}, "simmim": {}, "mae": {}, "siamim": {}, "simclr": {}}

@app.get("/", response_class=HTMLResponse)
@app.post("/", response_class=HTMLResponse)
async def index(
    request: Request,
    query_id: str = Form(None),
    query_index: int = Form(0),
    models: list = Form(["dino", "simmim"]),
    show_only_correct: bool = Form(False),
    view_mode: str = Form("full")
):
    queries = list(benchmark.keys())
    # Use query_id if provided, otherwise use query_index
    if query_id and query_id in queries:
        selected_query = query_id
        query_index = queries.index(query_id)
    else:
        query_index = max(0, min(query_index, len(queries) - 1))  # Ensure valid index
        selected_query = queries[query_index] if queries else ""
    logger.debug(f"Query index: {query_index}, Selected query: {selected_query}")
    logger.debug(f"Selected models: {models}")
    logger.debug(f"View mode: {view_mode}")

    # Limit to first two selected models for display
    display_models = models[:2]
    results = {}
    for model in display_models:
        if model in models_data:
            query_key = f"{selected_query}_hair.png"
            model_result = models_data[model].get(query_key, [])
            ground_truth = benchmark.get(selected_query, [])
            hits = [img for img in model_result if img.replace('_hair.png', '.jpg') in ground_truth]
            misses = [img for img in model_result if img not in hits]
            images = hits if show_only_correct else model_result
            results[model] = {
                "top100": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in images],
                "ground_truth": [f"/face_images/{img}" for img in ground_truth],
                "hits": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in hits],
                "misses": [{"hair": f"/hair_images/{img}", "face": f"/face_images/{img.replace('_hair.png', '.jpg')}"} for img in misses],
            }
            logger.debug(f"Model {model} - Query: {query_key}")
            logger.debug(f"Model {model} - Ground truth: {ground_truth}")
            logger.debug(f"Model {model} - Top100 (first 5): {images[:5]}")
            logger.debug(f"Model {model} - Hits: {hits}")
            logger.debug(f"Model {model} - Misses (first 5): {misses[:5]}")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "queries": queries,
            "query_index": query_index,
            "models": list(models_data.keys()),
            "selected_query": selected_query,
            "selected_models": models,
            "display_models": display_models,
            "show_only_correct": show_only_correct,
            "view_mode": view_mode,
            "results": results
        }
    )