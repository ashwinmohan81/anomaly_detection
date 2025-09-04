# Seamless Model Hosting Guide

## 🚀 Deployment Options Overview

| Option | Complexity | Cost | Scalability | Best For |
|--------|------------|------|-------------|----------|
| **Docker + Cloud** | Low | $ | Medium | Quick deployment |
| **AWS/GCP/Azure** | Medium | $$ | High | Production |
| **Kubernetes** | High | $$$ | Very High | Enterprise |
| **Serverless** | Medium | $ | Auto | Variable load |
| **Edge Computing** | Medium | $$ | High | Real-time |

## 1. 🐳 Docker + Cloud (Recommended for Start)

### Dockerfile Optimization
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Local Development
```yaml
version: '3.8'
services:
  anomaly-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - anomaly-api
    restart: unless-stopped
```

## 2. ☁️ Cloud Platform Deployments

### AWS Deployment (ECS + Fargate)
```yaml
# ecs-task-definition.json
{
  "family": "anomaly-detection-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "anomaly-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/anomaly-detection:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/anomaly-detection",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### Google Cloud Run
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/anomaly-detection:$COMMIT_SHA', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/anomaly-detection:$COMMIT_SHA']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: [
      'run', 'deploy', 'anomaly-detection',
      '--image', 'gcr.io/$PROJECT_ID/anomaly-detection:$COMMIT_SHA',
      '--region', 'us-central1',
      '--platform', 'managed',
      '--allow-unauthenticated',
      '--memory', '2Gi',
      '--cpu', '2',
      '--max-instances', '10'
    ]
```

### Azure Container Instances
```yaml
# azure-deploy.yaml
apiVersion: 2018-10-01
location: eastus
name: anomaly-detection
properties:
  containers:
  - name: anomaly-api
    properties:
      image: your-registry.azurecr.io/anomaly-detection:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
      ports:
      - port: 8000
        protocol: TCP
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
```

## 3. 🚀 Serverless Deployment

### AWS Lambda + API Gateway
```python
# lambda_handler.py
import json
from mangum import Mangum
from main import app

# Wrap FastAPI app for Lambda
handler = Mangum(app, lifespan="off")

def lambda_handler(event, context):
    return handler(event, context)
```

```yaml
# serverless.yml
service: anomaly-detection-api

provider:
  name: aws
  runtime: python3.9
  region: us-east-1
  memorySize: 3008
  timeout: 30
  environment:
    ENVIRONMENT: production

functions:
  api:
    handler: lambda_handler.lambda_handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true
      - http:
          path: /
          method: ANY
          cors: true

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
    slim: true
```

### Vercel Deployment
```json
{
  "version": 2,
  "builds": [
    {
      "src": "main.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "main.py"
    }
  ]
}
```

## 4. ☸️ Kubernetes Deployment

### Kubernetes Manifests
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-detection-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: anomaly-detection-api
  template:
    metadata:
      labels:
        app: anomaly-detection-api
    spec:
      containers:
      - name: api
        image: your-registry/anomaly-detection:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: anomaly-detection-service
spec:
  selector:
    app: anomaly-detection-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: anomaly-detection-ingress
spec:
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: anomaly-detection-service
            port:
              number: 80
```

### Helm Chart
```yaml
# Chart.yaml
apiVersion: v2
name: anomaly-detection
description: Anomaly Detection API
version: 0.1.0
appVersion: "1.0.0"

# values.yaml
replicaCount: 3

image:
  repository: your-registry/anomaly-detection
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: anomaly-detection-tls
      hosts:
        - api.yourdomain.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

## 5. 🔧 Production Optimizations

### Enhanced FastAPI App
```python
# main_production.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
models_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up anomaly detection API...")
    # Pre-load models, initialize connections, etc.
    yield
    # Shutdown
    logger.info("Shutting down anomaly detection API...")

app = FastAPI(
    title="Anomaly Detection API",
    description="Production-ready anomaly detection service",
    version="1.0.0",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["api.yourdomain.com", "*.yourdomain.com"]
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Rate limiting (using slowapi)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Your existing endpoints with rate limiting...
```

### Nginx Configuration
```nginx
# nginx.conf
upstream anomaly_api {
    server anomaly-api:8000;
    server anomaly-api-2:8000;
    server anomaly-api-3:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://anomaly_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://anomaly_api/health;
    }
}
```

## 6. 📊 Monitoring & Observability

### Prometheus Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_MODELS = Gauge('active_models_total', 'Number of active models')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')

@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Docker Compose with Monitoring
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  anomaly-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: anomaly_detection
      POSTGRES_USER: api_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - anomaly-api
    restart: unless-stopped

volumes:
  postgres_data:
  grafana_data:
```

## 7. 🚀 Quick Start Commands

### Local Development
```bash
# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f anomaly-api

# Scale up
docker-compose up -d --scale anomaly-api=3
```

### Cloud Deployment
```bash
# AWS ECS
aws ecs create-cluster --cluster-name anomaly-detection
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
aws ecs create-service --cluster anomaly-detection --service-name api --task-definition anomaly-detection-api

# Google Cloud Run
gcloud run deploy anomaly-detection --source . --platform managed --region us-central1

# Azure Container Instances
az container create --resource-group myResourceGroup --name anomaly-api --image your-registry.azurecr.io/anomaly-detection:latest
```

### Kubernetes
```bash
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -l app=anomaly-detection-api

# Scale up
kubectl scale deployment anomaly-detection-api --replicas=5
```

## 8. 💰 Cost Optimization

### Resource Sizing
```yaml
# Small deployment (development)
resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi

# Medium deployment (production)
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 1000m
    memory: 2Gi

# Large deployment (high traffic)
resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

### Auto-scaling
```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: anomaly-detection-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-detection-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🎯 Recommended Deployment Strategy

### Phase 1: Quick Start (Week 1)
1. **Docker + Cloud Run/AWS Fargate**
2. **Basic monitoring** (health checks)
3. **Single region deployment**

### Phase 2: Production Ready (Week 2-3)
1. **Kubernetes cluster**
2. **Full monitoring stack** (Prometheus + Grafana)
3. **Load balancing + SSL**
4. **Auto-scaling**

### Phase 3: Enterprise (Month 2+)
1. **Multi-region deployment**
2. **Advanced monitoring** (distributed tracing)
3. **CI/CD pipeline**
4. **Disaster recovery**

This gives you a complete roadmap for seamlessly hosting your anomaly detection models at any scale! 🚀
