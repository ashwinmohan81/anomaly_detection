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

from generic_anomaly_detector import GenericAnomalyDetector

def serialize_dates(obj):
    """Custom JSON serializer for datetime objects."""
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).strftime('%Y-%m-%d %H:%M:%S')
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Anomaly Detection API",
    description="Universal anomaly detection service for any business keys and attributes",
    version="2.0.0"
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
class GenericTrainingRequest(BaseModel):
    algorithm: str = "isolation_forest"
    contamination: float = 0.1
    business_key: str  # Column that identifies business entities
    target_attributes: List[str]  # List of columns to detect anomalies in
    feature_columns: Optional[List[str]] = None
    time_column: Optional[str] = None  # Optional time column for temporal features
    anomaly_labels: Optional[str] = None  # For supervised learning

class GenericPredictionRequest(BaseModel):
    model_id: str
    data: Dict[str, Any]

class GenericBatchPredictionRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]
    include_predictions: Optional[bool] = True  # Whether to include full predictions array
    include_scores: Optional[bool] = True      # Whether to include full scores array
    page: Optional[int] = 1                    # Page number for pagination
    page_size: Optional[int] = 100             # Number of records per page
    summary_only: Optional[bool] = False       # Return only summary statistics

class GenericModelInfo(BaseModel):
    model_id: str
    algorithm: str
    contamination: float
    business_key: str
    target_attributes: List[str]
    feature_columns: List[str]
    time_column: Optional[str]
    anomaly_labels: Optional[str]
    training_stats: Dict[str, Any]
    created_at: str

@app.get("/")
async def root():
    return {"message": "Anomaly Detection API", "version": "2.0.0"}

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
        
        # Convert any datetime columns to strings for JSON serialization
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
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
async def train_model(dataset_id: str, request: GenericTrainingRequest):
    """Train a generic anomaly detection model."""
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
        
        # Validate required columns
        required_cols = [request.business_key] + request.target_attributes
        if request.anomaly_labels:
            required_cols.append(request.anomaly_labels)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Create and train model
        detector = GenericAnomalyDetector(
            algorithm=request.algorithm,
            contamination=request.contamination
        )
        
        detector.train(
            df=df,
            business_key=request.business_key,
            target_attributes=request.target_attributes,
            feature_columns=request.feature_columns,
            time_column=request.time_column,
            anomaly_labels=request.anomaly_labels
        )
        
        # Generate model ID
        model_id = f"generic_{request.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save model
        model_path = f"{model_id}.pkl"
        detector.save_model(model_path)
        
        # Store model info
        models[model_id] = detector
        model_metadata[model_id] = {
            "model_id": model_id,
            "algorithm": request.algorithm,
            "contamination": request.contamination,
            "business_key": request.business_key,
            "target_attributes": request.target_attributes,
            "feature_columns": detector.feature_columns,
            "time_column": request.time_column,
            "anomaly_labels": request.anomaly_labels,
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
async def predict_anomaly(request: GenericPredictionRequest):
    """Predict anomaly for a single data point."""
    try:
        if request.model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        detector = models[request.model_id]
        metadata = model_metadata[request.model_id]
        
        # Convert single data point to DataFrame
        df = pd.DataFrame([request.data])
        result = detector.predict(
            df=df, 
            business_key=metadata['business_key'], 
            time_column=metadata['time_column']
        )
        
        # Get entity ID from the input data
        entity_id = request.data[metadata['business_key']]
        
        # Enhance prediction_analysis with detailed predictions and anomaly indicators
        enhanced_prediction_analysis = result['prediction_analysis'].copy()
        enhanced_prediction_analysis[entity_id].update({
            "predictions": [result['predictions'][0]],
            "anomaly_indicators": [result['anomaly_indicators'][0]]
        })
        
        return {
            "model_id": request.model_id,
            "anomaly_count": result['anomaly_count'],
            "anomaly_rate": result['anomaly_rate'],
            "prediction_analysis": enhanced_prediction_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(request: GenericBatchPredictionRequest):
    """Predict anomalies for multiple data points with pagination and summary options."""
    try:
        if request.model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        detector = models[request.model_id]
        metadata = model_metadata[request.model_id]
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        result = detector.predict(
            df=df, 
            business_key=metadata['business_key'], 
            time_column=metadata['time_column']
        )
        
        # Prepare base response
        response = {
            "model_id": request.model_id,
            "timestamp": datetime.now().isoformat(),
            "total_records": len(request.data),
            "anomaly_count": result['anomaly_count'],
            "anomaly_rate": result['anomaly_rate']
        }
        
        # Handle summary_only option
        if request.summary_only:
            response["prediction_analysis"] = result['prediction_analysis']
            return response
        
        # Handle pagination and inclusion options
        predictions = result['predictions']
        anomaly_indicators = result['anomaly_indicators']
        
        # Calculate pagination
        total_records = len(predictions)
        start_idx = (request.page - 1) * request.page_size
        end_idx = min(start_idx + request.page_size, total_records)
        
        # Add pagination info
        response.update({
            "pagination": {
                "page": request.page,
                "page_size": request.page_size,
                "total_pages": (total_records + request.page_size - 1) // request.page_size,
                "has_next": end_idx < total_records,
                "has_previous": request.page > 1
            }
        })
        
        # Group predictions and anomaly indicators by entity ID
        entity_predictions = {}
        entity_indicators = {}
        
        for i, (prediction, indicator) in enumerate(zip(predictions, anomaly_indicators)):
            entity_id = df.iloc[i][metadata['business_key']]
            
            if entity_id not in entity_predictions:
                entity_predictions[entity_id] = []
                entity_indicators[entity_id] = []
            
            entity_predictions[entity_id].append(prediction)
            entity_indicators[entity_id].append(indicator)
        
        # Enhance prediction_analysis with detailed predictions and anomaly indicators
        enhanced_prediction_analysis = result['prediction_analysis'].copy()
        
        for entity_id in enhanced_prediction_analysis:
            if request.include_predictions and entity_id in entity_predictions:
                enhanced_prediction_analysis[entity_id]["predictions"] = entity_predictions[entity_id]
            else:
                enhanced_prediction_analysis[entity_id]["predictions"] = None
                
            if request.include_scores and entity_id in entity_indicators:
                enhanced_prediction_analysis[entity_id]["anomaly_indicators"] = entity_indicators[entity_id]
            else:
                enhanced_prediction_analysis[entity_id]["anomaly_indicators"] = None
        
        response["prediction_analysis"] = enhanced_prediction_analysis
        
        return response
        
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
        
        detector = GenericAnomalyDetector.load_model(model_path)
        models[model_id] = detector
        
        # Update metadata
        model_metadata[model_id] = {
            "model_id": model_id,
            "algorithm": detector.algorithm,
            "contamination": detector.contamination,
            "business_key": detector.business_keys[0] if detector.business_keys else "unknown",
            "target_attributes": detector.target_attributes,
            "feature_columns": detector.feature_columns,
            "time_column": getattr(detector, 'time_column', None),
            "anomaly_labels": getattr(detector, 'anomaly_labels', None),
            "training_stats": detector.training_stats,
            "created_at": datetime.now().isoformat(),
            "model_path": model_path
        }
        
        return {"message": f"Model {model_id} loaded successfully"}
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/algorithms")
async def list_algorithms():
    """List available algorithms."""
    return {
        "algorithms": [
            {
                "name": "isolation_forest",
                "description": "Isolation Forest - Unsupervised anomaly detection",
                "type": "unsupervised"
            },
            {
                "name": "one_class_svm",
                "description": "One-Class SVM - Unsupervised anomaly detection",
                "type": "unsupervised"
            },
            {
                "name": "local_outlier_factor",
                "description": "Local Outlier Factor - Unsupervised anomaly detection",
                "type": "unsupervised"
            },
            {
                "name": "random_forest",
                "description": "Random Forest - Supervised anomaly detection",
                "type": "supervised"
            },
            {
                "name": "logistic_regression",
                "description": "Logistic Regression - Supervised anomaly detection",
                "type": "supervised"
            }
        ]
    }

@app.get("/examples")
async def get_examples():
    """Get example use cases and data formats."""
    return {
        "examples": [
            {
                "name": "Fund Rating Anomaly Detection",
                "description": "Detect anomalies in fund ratings and values",
                "business_key": "fund_id",
                "target_attributes": ["fund_rating", "fund_value", "fund_size", "fund_returns"],
                "time_column": "date",
                "anomaly_labels": "is_anomaly"
            },
            {
                "name": "Stock Price Anomaly Detection",
                "description": "Detect anomalies in stock prices and volumes",
                "business_key": "symbol",
                "target_attributes": ["price", "volume", "returns"],
                "time_column": "date",
                "anomaly_labels": None
            },
            {
                "name": "Customer Behavior Anomaly Detection",
                "description": "Detect anomalies in customer spending patterns",
                "business_key": "customer_id",
                "target_attributes": ["spending", "frequency", "amount"],
                "time_column": "transaction_date",
                "anomaly_labels": "fraud_flag"
            },
            {
                "name": "Sensor Data Anomaly Detection",
                "description": "Detect anomalies in IoT sensor readings",
                "business_key": "sensor_id",
                "target_attributes": ["temperature", "humidity", "pressure"],
                "time_column": "timestamp",
                "anomaly_labels": None
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
