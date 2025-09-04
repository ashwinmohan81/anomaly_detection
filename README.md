# Anomaly Detection API

A generic anomaly detection service that can train models on any dataset and serve predictions via REST API.

## Features

- **Multiple Algorithms**: Isolation Forest, One-Class SVM, Local Outlier Factor, Random Forest, Logistic Regression
- **Supervised & Unsupervised Learning**: Train with labeled data for better performance
- **Automatic Feature Engineering**: Lag features, rolling statistics, time-based features
- **Flexible Data Input**: CSV, JSON, Parquet files
- **REST API**: FastAPI-based service with comprehensive endpoints
- **Model Management**: Save, load, and version models
- **Batch & Single Predictions**: Support for both real-time and batch processing

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Run Examples

```bash
# Unsupervised learning example
python example_usage.py

# Supervised learning example (with labeled data)
python supervised_example.py
```

## API Endpoints

### Core Endpoints

- `POST /upload-dataset` - Upload and validate a dataset
- `POST /train/{dataset_id}` - Train an anomaly detection model
- `POST /predict` - Single prediction
- `POST /predict-batch` - Batch prediction
- `GET /models` - List all models
- `GET /models/{model_id}` - Get model information
- `DELETE /models/{model_id}` - Delete a model

### Model Management Endpoints

- `POST /models/{model_id}/load` - Load a model from disk
- `POST /models/{model_id}/retrain/{dataset_id}` - Retrain existing model with new data
- `POST /models/{model_id}/combine/{dataset_id}` - Combine datasets for retraining

### Utility Endpoints

- `GET /` - API information
- `GET /health` - Health check

## Usage Examples

### 1. Upload Dataset

```python
import requests

# Upload CSV file
with open('data.csv', 'rb') as f:
    files = {'file': ('data.csv', f, 'text/csv')}
    response = requests.post('http://localhost:8000/upload-dataset', files=files)
    dataset_id = response.json()['dataset_id']
```

### 2. Train Model

**Unsupervised Learning:**
```python
training_request = {
    "algorithm": "isolation_forest",
    "contamination": 0.1,
    "target_column": "price",
    "feature_columns": ["volume", "amount"]  # Optional
}

response = requests.post(
    f'http://localhost:8000/train/{dataset_id}',
    json=training_request
)
model_id = response.json()['model_id']
```

**Supervised Learning (with labeled data):**
```python
training_request = {
    "algorithm": "random_forest",  # Use supervised algorithm
    "contamination": 0.1,
    "target_column": "price",
    "feature_columns": ["volume", "amount"],
    "label_column": "is_anomaly"  # Column with 0/1 labels
}

response = requests.post(
    f'http://localhost:8000/train/{dataset_id}',
    json=training_request
)
model_id = response.json()['model_id']
```

### 3. Make Predictions

```python
# Single prediction
prediction_request = {
    "model_id": model_id,
    "data": {
        "price": 120.5,
        "volume": 950,
        "amount": 10000
    }
}

response = requests.post('http://localhost:8000/predict', json=prediction_request)
result = response.json()
print(f"Is anomaly: {result['prediction']['is_anomaly']}")
```

### 4. Batch Prediction

```python
batch_request = {
    "model_id": model_id,
    "data": [
        {"price": 120.5, "volume": 950, "amount": 10000},
        {"price": 200.0, "volume": 950, "amount": 10000},  # Likely anomaly
        {"price": 118.2, "volume": 1100, "amount": 12000}
    ]
}

response = requests.post('http://localhost:8000/predict-batch', json=batch_request)
result = response.json()
print(f"Anomalies found: {result['predictions']['anomaly_count']}")
```

## Supported Algorithms

### Unsupervised Algorithms

#### 1. Isolation Forest
- **Best for**: High-dimensional data, mixed data types
- **Pros**: Fast, handles outliers well, no assumptions about data distribution
- **Cons**: May struggle with local anomalies

#### 2. One-Class SVM
- **Best for**: Complex non-linear patterns
- **Pros**: Good for complex boundaries, kernel-based
- **Cons**: Slower on large datasets, sensitive to hyperparameters

#### 3. Local Outlier Factor
- **Best for**: Local density-based anomalies
- **Pros**: Good for local anomalies, density-based
- **Cons**: Computationally expensive, sensitive to k parameter

### Supervised Algorithms

#### 4. Random Forest
- **Best for**: When you have labeled anomaly data
- **Pros**: High accuracy, handles mixed data types, feature importance
- **Cons**: Requires labeled data, can overfit

#### 5. Logistic Regression
- **Best for**: Linear relationships, interpretable models
- **Pros**: Fast, interpretable, good baseline
- **Cons**: Assumes linear relationships, requires labeled data

## Data Requirements

### Input Format
- **CSV**: Standard comma-separated values
- **JSON**: Array of objects or JSON lines
- **Parquet**: Columnar format for large datasets

### Required Columns
- At least one numeric column for the target variable
- Additional numeric columns for features (optional)

### Automatic Feature Engineering
The system automatically creates:
- **Lag features**: Previous values (1, 2, 3, 7 periods)
- **Rolling statistics**: Moving averages and standard deviations
- **Z-scores**: Normalized values relative to rolling mean/std
- **Time features**: Hour, day of week, month (if date column detected)

## Model Management

### Saving Models
Models are automatically saved when trained and can be loaded later:

```python
# Load existing model
response = requests.post(f'http://localhost:8000/models/{model_id}/load')
```

### Model Information
Get detailed information about any model:

```python
response = requests.get(f'http://localhost:8000/models/{model_id}')
model_info = response.json()
print(f"Algorithm: {model_info['algorithm']}")
print(f"Training stats: {model_info['training_stats']}")
```

### Retraining Models
Retrain existing models with additional data:

```python
# Upload new data
new_dataset_id = upload_dataset("new_data.csv")

# Retrain existing model
retrain_request = {
    "algorithm": "isolation_forest",
    "contamination": 0.1,
    "target_column": "price"
}

response = requests.post(
    f'http://localhost:8000/models/{model_id}/retrain/{new_dataset_id}',
    json=retrain_request
)
```

### Combining Datasets
Combine multiple datasets before training:

```python
# Combine datasets
response = requests.post(
    f'http://localhost:8000/models/{model_id}/combine/{new_dataset_id}'
)
combined_id = response.json()['combined_dataset_id']

# Train on combined data
response = requests.post(
    f'http://localhost:8000/train/{combined_id}',
    json=training_request
)
```

## Configuration

### Training Parameters

- **algorithm**: `isolation_forest`, `one_class_svm`, `local_outlier_factor`, `random_forest`, `logistic_regression`
- **contamination**: Expected proportion of anomalies (0.01 to 0.5)
- **target_column**: Column to detect anomalies in
- **feature_columns**: Optional list of feature columns
- **label_column**: Optional column with true anomaly labels (0=normal, 1=anomaly) for supervised learning

### API Configuration

- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8000
- **CORS**: Enabled for all origins
- **Logging**: INFO level

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
export MODEL_STORAGE_PATH=/app/models
export LOG_LEVEL=INFO
export MAX_UPLOAD_SIZE=100MB
```

## Monitoring

### Health Checks
```bash
curl http://localhost:8000/health
```

### Model Performance
```python
# Get model statistics
response = requests.get(f'http://localhost:8000/models/{model_id}')
stats = response.json()['training_stats']
print(f"Anomaly rate: {stats['anomaly_rate']:.2%}")
```

## Troubleshooting

### Common Issues

1. **"Model not found"**: Ensure model is trained and loaded
2. **"Dataset not found"**: Check dataset_id is correct
3. **"Target column not found"**: Verify column name matches exactly
4. **Memory issues**: Use smaller datasets or increase server memory

### Logs
Check application logs for detailed error information:
```bash
uvicorn main:app --reload --log-level debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
