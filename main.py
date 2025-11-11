"""
GDPulse - Local Backend API
Orchestrates requests to HF Space and provides caching

This module handles:
- Request validation and forwarding to HF Space
- Response caching (15-minute TTL)
- CORS for React frontend
- Health checks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json

app = FastAPI(title="GDPulse Backend", version="1.0.0")

# Allow requests from frontend
origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:5173",  # Vite default port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face Space configuration
# Public URL: https://huggingface.co/spaces/Koan39/GDPulse
HF_SPACE = "Koan39/GDPulse"

# Simple in-memory cache
CACHE: Dict[str, Dict] = {}
CACHE_TTL = 15 * 60  # 15 minutes in seconds


# Pydantic Models
class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    std_error: float


class PredictionResponse(BaseModel):
    model_name: str
    prediction: float
    confidence_interval: ConfidenceInterval
    actual_gdp: Optional[float] = None
    date: str
    timestamp: str
    model_description: str
    cached: bool = False


class AllPredictionsResponse(BaseModel):
    predictions: List[PredictionResponse]
    timestamp: str
    cached: bool = False


# Cache management
def get_cache_key(model_name: str, features: Optional[List[float]] = None) -> str:
    """Generate cache key for a prediction request."""
    if features is None:
        return f"{model_name}:default"
    return f"{model_name}:{hash(tuple(features))}"


def get_cached_prediction(cache_key: str) -> Optional[Dict]:
    """Get prediction from cache if available and not expired."""
    if cache_key not in CACHE:
        return None

    cached_data = CACHE[cache_key]
    age = (datetime.utcnow() - cached_data["cached_at"]).total_seconds()

    if age > CACHE_TTL:
        del CACHE[cache_key]
        return None

    return cached_data["response"]


def cache_prediction(cache_key: str, response: dict):
    """Store prediction in cache."""
    CACHE[cache_key] = {
        "response": response,
        "cached_at": datetime.utcnow()
    }


# HF Space communication
async def call_hf_space(model_name: str, features: Optional[List[float]] = None) -> dict:
    """
    Call the HF Space /predict endpoint.
    Runs in thread to avoid blocking async event loop.
    """
    try:
        client = Client(HF_SPACE)

        # Prepare request payload
        request_payload = {
            "model_name": model_name,
            "features": features
        }

        # Call HF Space API (running in thread)
        result = await asyncio.to_thread(
            client.predict,
            model_name,
            features,
            api_name="/predict"
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error calling HF Space ({HF_SPACE}): {str(e)}"
        )


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "GDPulse Backend",
        "status": "running",
        "version": "1.0.0",
        "hf_space": HF_SPACE
    }


@app.get("/health")
async def health_check():
    """Check health of backend and HF Space."""
    try:
        client = Client(HF_SPACE)
        hf_status = "connected"
    except Exception as e:
        hf_status = f"disconnected: {str(e)}"

    return {
        "backend_status": "healthy",
        "hf_space_status": hf_status,
        "cache_size": len(CACHE),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(model_name: str, features: Optional[List[float]] = None):
    """
    Get a GDP prediction for a specific model.

    Parameters:
    - model_name: One of 'nowcasting_h1', 'forecasting_h1', 'forecasting_h2', 'forecasting_h3'
    - features: Optional list of feature values

    Returns cached response if available (within 15 minutes), otherwise calls HF Space.
    """
    # Check cache
    cache_key = get_cache_key(model_name, features)
    cached_response = get_cached_prediction(cache_key)

    if cached_response:
        cached_response["cached"] = True
        return cached_response

    # Call HF Space
    result = await call_hf_space(model_name, features)

    # Parse response - ensure it's in the right format
    if isinstance(result, str):
        result = json.loads(result)

    result["cached"] = False

    # Cache the result
    cache_prediction(cache_key, result)

    return result


@app.post("/predict-all", response_model=AllPredictionsResponse)
async def predict_all(features: Optional[List[float]] = None):
    """
    Get predictions from all 4 models at once.

    Parameters:
    - features: Optional list of feature values (used for all models)

    Returns all model predictions in a single response.
    """
    models = [
        "nowcasting_h1",
        "forecasting_h1",
        "forecasting_h2",
        "forecasting_h3"
    ]

    predictions = []
    cached_count = 0

    for model_name in models:
        cache_key = get_cache_key(model_name, features)
        cached_response = get_cached_prediction(cache_key)

        if cached_response:
            predictions.append(cached_response)
            cached_count += 1
        else:
            result = await call_hf_space(model_name, features)
            if isinstance(result, str):
                result = json.loads(result)
            result["cached"] = False
            predictions.append(result)
            cache_prediction(cache_key, result)

    return AllPredictionsResponse(
        predictions=predictions,
        timestamp=datetime.utcnow().isoformat(),
        cached=(cached_count == len(models))
    )


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached predictions."""
    cache_size = len(CACHE)
    CACHE.clear()
    return {
        "message": "Cache cleared",
        "previous_size": cache_size,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/cache/info")
async def cache_info():
    """Get information about current cache."""
    entries = []
    for key, data in CACHE.items():
        age = (datetime.utcnow() - data["cached_at"]).total_seconds()
        entries.append({
            "key": key,
            "age_seconds": age,
            "ttl_remaining": CACHE_TTL - age
        })

    return {
        "total_entries": len(CACHE),
        "entries": entries,
        "ttl_seconds": CACHE_TTL
    }
