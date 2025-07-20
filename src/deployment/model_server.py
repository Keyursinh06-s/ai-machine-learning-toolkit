"""
Model Deployment Server
FastAPI-based server for serving ML models in production
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import pickle
import io
from PIL import Image
import uvicorn
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model Server",
    description="Production-ready ML model serving API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {}

class PredictionRequest(BaseModel):
    features: List[float]
    model_name: str

class PredictionResponse(BaseModel):
    prediction: float
    confidence: Optional[float] = None
    model_name: str

class ModelInfo(BaseModel):
    name: str
    type: str
    input_shape: List[int]
    output_shape: List[int]
    loaded: bool

@app.on_startup
async def startup_event():
    """Load models on startup"""
    logger.info("Starting ML Model Server...")
    # Load default models here
    await load_default_models()

async def load_default_models():
    """Load pre-trained models"""
    try:
        # Example: Load a simple sklearn model
        # models['example_classifier'] = pickle.load(open('models/classifier.pkl', 'rb'))
        logger.info("Default models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading default models: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "ML Model Server is running", "status": "healthy"}

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all loaded models"""
    model_list = []
    for name, model in models.items():
        model_info = ModelInfo(
            name=name,
            type=type(model).__name__,
            input_shape=[1, 10],  # Example shape
            output_shape=[1],
            loaded=True
        )
        model_list.append(model_info)
    return model_list

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using loaded models"""
    try:
        if request.model_name not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = models[request.model_name]
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            prediction = model.predict(features)[0]
            confidence = max(model.predict_proba(features)[0])
        else:
            prediction = model.predict(features)[0]
            confidence = None
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=confidence,
            model_name=request.model_name
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), model_name: str = "image_classifier"):
    """Image classification endpoint"""
    try:
        if model_name not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image (example)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        
        # Make prediction
        model = models[model_name]
        prediction = model.predict(image_array.reshape(1, -1))
        
        return {
            "prediction": float(prediction[0]),
            "model_name": model_name,
            "image_size": image.size
        }
    
    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/load")
async def load_model(model_name: str, model_path: str):
    """Load a model from file"""
    try:
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_path.endswith('.pth'):
            model = torch.load(model_path, map_location='cpu')
        else:
            raise HTTPException(status_code=400, detail="Unsupported model format")
        
        models[model_name] = model
        logger.info(f"Model {model_name} loaded successfully")
        
        return {"message": f"Model {model_name} loaded successfully"}
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from memory"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del models[model_name]
    logger.info(f"Model {model_name} unloaded")
    
    return {"message": f"Model {model_name} unloaded successfully"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "memory_usage": "N/A",  # Could add actual memory monitoring
        "uptime": "N/A"
    }

if __name__ == "__main__":
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )