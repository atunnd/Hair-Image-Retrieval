
import json
from pathlib import Path
from fastapi import HTTPException

def load_json(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in file: {path}")

def load_model_results(file_path: str):
    raw = load_json(file_path)
    return {entry["query"]: entry["top100"] for entry in raw}

def load_benchmark(file_path: str):
    data = load_json(file_path)
    return {entry["query_image"].replace(".jpg", ""): entry["ground_truth"] for entry in data}
