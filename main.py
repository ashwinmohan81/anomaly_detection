from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import os
import logging
from datetime import datetime
import json

from anomaly_detector import AnomalyDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Anomaly Detection API",
    description="Generic anomaly detection service with multiple algorithms",
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
model_metadata = {}

# Pydantic models for API
class TrainingRequest(BaseModel):
    algorithm: str = "isolation_forest"
    contamination: float = 0.1
    target_column: str
    feature_columns: Optional[List[str]] = None
    label_column: Optional[str] = None  # For supervised learning

class PredictionRequest(BaseModel):
    model_id: str
    data: Dict[str, Any]

class BatchPredictionRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]

class ModelInfo(BaseModel):
    model_id: str
    algorithm: str
    contamination: float
    target_column: str
    feature_columns: List[str]
    training_stats: Dict[str, Any]
    created_at: str

@app.get("/")
async def root():
    return {"message": "Anomaly Detection API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and validate a dataset."""
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type and read
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.parquet'):
            try:
                df = pd.read_parquet(io.BytesIO(content))
            except Exception as e:
                # Fallback to CSV if parquet fails
                raise HTTPException(status_code=400, detail=f"Parquet reading failed: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Basic data validation
        data_info = {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Store dataset temporarily (in production, use proper storage)
        dataset_id = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            df.to_parquet(f"temp_{dataset_id}.parquet")
        except Exception as e:
            # Fallback to CSV if parquet fails
            df.to_csv(f"temp_{dataset_id}.csv", index=False)
        
        return {
            "dataset_id": dataset_id,
            "data_info": data_info,
            "message": "Dataset uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")

@app.post("/train/{dataset_id}")
async def train_model(dataset_id: str, request: TrainingRequest):
    """Train an anomaly detection model."""
    try:
        # Load dataset
        dataset_path_parquet = f"temp_{dataset_id}.parquet"
        dataset_path_csv = f"temp_{dataset_id}.csv"
        
        if os.path.exists(dataset_path_parquet):
            df = pd.read_parquet(dataset_path_parquet)
        elif os.path.exists(dataset_path_csv):
            df = pd.read_csv(dataset_path_csv)
        else:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Validate target column
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{request.target_column}' not found in dataset"
            )
        
        # Create and train model
        detector = AnomalyDetector(
            algorithm=request.algorithm,
            contamination=request.contamination
        )
        
        detector.train(
            df=df,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            label_column=request.label_column
        )
        
        # Generate model ID
        model_id = f"model_{request.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model
        model_path = f"{model_id}.pkl"
        detector.save_model(model_path)
        
        # Store model info
        models[model_id] = detector
        model_metadata[model_id] = {
            "model_id": model_id,
            "algorithm": request.algorithm,
            "contamination": request.contamination,
            "target_column": request.target_column,
            "feature_columns": detector.feature_columns,
            "training_stats": detector.training_stats,
            "created_at": datetime.now().isoformat(),
            "model_path": model_path
        }
        
        # Clean up temporary dataset
        if os.path.exists(dataset_path_parquet):
            os.remove(dataset_path_parquet)
        elif os.path.exists(dataset_path_csv):
            os.remove(dataset_path_csv)
        
        return {
            "model_id": model_id,
            "training_stats": detector.training_stats,
            "message": "Model trained successfully"
        }
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/predict")
async def predict_anomaly(request: PredictionRequest):
    """Predict anomaly for a single data point."""
    try:
        if request.model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        detector = models[request.model_id]
        result = detector.predict_single(request.data)
        
        return {
            "model_id": request.model_id,
            "prediction": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict anomalies for multiple data points."""
    try:
        if request.model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        detector = models[request.model_id]
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        result = detector.predict(df)
        
        return {
            "model_id": request.model_id,
            "predictions": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making batch prediction: {str(e)}")

@app.get("/models")
async def list_models():
    """List all trained models."""
    return {
        "models": list(model_metadata.values()),
        "count": len(models)
    }

@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model."""
    if model_id not in model_metadata:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model_metadata[model_id]

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model."""
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Remove model file
    model_path = model_metadata[model_id]["model_path"]
    if os.path.exists(model_path):
        os.remove(model_path)
    
    # Remove from memory
    del models[model_id]
    del model_metadata[model_id]
    
    return {"message": f"Model {model_id} deleted successfully"}

@app.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """Load a model from disk."""
    try:
        model_path = f"{model_id}.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        detector = AnomalyDetector.load_model(model_path)
        models[model_id] = detector
        
        # Update metadata
        model_metadata[model_id] = {
            "model_id": model_id,
            "algorithm": detector.algorithm,
            "contamination": detector.contamination,
            "target_column": detector.target_column,
            "feature_columns": detector.feature_columns,
            "training_stats": detector.training_stats,
            "created_at": datetime.now().isoformat(),
            "model_path": model_path
        }
        
        return {"message": f"Model {model_id} loaded successfully"}
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/models/{model_id}/retrain/{dataset_id}")
async def retrain_model(model_id: str, dataset_id: str, request: TrainingRequest):
    """Retrain an existing model with additional data."""
    try:
        # Load existing model
        if model_id not in models:
            # Try to load from disk
            model_path = f"{model_id}.pkl"
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail="Model not found")
            detector = AnomalyDetector.load_model(model_path)
        else:
            detector = models[model_id]
        
        # Load new dataset
        dataset_path = f"temp_{dataset_id}.parquet"
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        new_df = pd.read_parquet(dataset_path)
        
        # Validate target column
        if request.target_column not in new_df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{request.target_column}' not found in dataset"
            )
        
        # Get original training data (if available)
        original_stats = detector.training_stats
        original_data_path = f"original_data_{model_id}.parquet"
        
        if os.path.exists(original_data_path):
            # Load original data
            original_df = pd.read_parquet(original_data_path)
            # Combine with new data
            combined_df = pd.concat([original_df, new_df], ignore_index=True)
            logger.info(f"Retraining with {len(original_df)} original + {len(new_df)} new records")
        else:
            # No original data available, use only new data
            combined_df = new_df
            logger.info(f"Retraining with {len(new_df)} new records (no original data found)")
        
        # Retrain model
        detector.train(
            df=combined_df,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            label_column=request.label_column
        )
        
        # Save updated model
        model_path = f"{model_id}.pkl"
        detector.save_model(model_path)
        
        # Save combined data for future retraining
        combined_df.to_parquet(original_data_path)
        
        # Update model info
        models[model_id] = detector
        model_metadata[model_id] = {
            "model_id": model_id,
            "algorithm": detector.algorithm,
            "contamination": detector.contamination,
            "target_column": detector.target_column,
            "feature_columns": detector.feature_columns,
            "training_stats": detector.training_stats,
            "created_at": datetime.now().isoformat(),
            "model_path": model_path,
            "retrained": True
        }
        
        # Clean up temporary dataset
        os.remove(dataset_path)
        
        return {
            "model_id": model_id,
            "training_stats": detector.training_stats,
            "message": "Model retrained successfully",
            "total_records": len(combined_df)
        }
        
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retraining model: {str(e)}")

@app.post("/models/{model_id}/combine/{dataset_id}")
async def combine_datasets(model_id: str, dataset_id: str):
    """Combine new dataset with existing model's training data."""
    try:
        # Load existing model
        if model_id not in models:
            model_path = f"{model_id}.pkl"
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail="Model not found")
            detector = AnomalyDetector.load_model(model_path)
        else:
            detector = models[model_id]
        
        # Load new dataset
        dataset_path = f"temp_{dataset_id}.parquet"
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        new_df = pd.read_parquet(dataset_path)
        
        # Get original training data
        original_data_path = f"original_data_{model_id}.parquet"
        
        if os.path.exists(original_data_path):
            original_df = pd.read_parquet(original_data_path)
            combined_df = pd.concat([original_df, new_df], ignore_index=True)
        else:
            # No original data, use only new data
            combined_df = new_df
        
        # Save combined dataset
        combined_dataset_id = f"combined_{model_id}_{dataset_id}"
        combined_path = f"temp_{combined_dataset_id}.parquet"
        combined_df.to_parquet(combined_path)
        
        # Clean up temporary dataset
        os.remove(dataset_path)
        
        return {
            "combined_dataset_id": combined_dataset_id,
            "total_records": len(combined_df),
            "original_records": len(original_df) if os.path.exists(original_data_path) else 0,
            "new_records": len(new_df),
            "message": "Datasets combined successfully"
        }
        
    except Exception as e:
        logger.error(f"Error combining datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error combining datasets: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
