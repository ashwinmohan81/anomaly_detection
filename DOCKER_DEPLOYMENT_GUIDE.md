# 🐳 Docker Deployment Guide

Complete guide for deploying the Anomaly Detection APIs using Docker.

## 🚀 **Quick Start**

### **1. Single API Deployment**
```bash
# Run Main API (Port 8000)
./docker-deploy.sh

# Run Generic API (Port 8001)
./docker-deploy.sh -a generic -p 8001

# Run with rebuild
./docker-deploy.sh -b -a main
```

### **2. Multi-Service Deployment**
```bash
# Run both APIs with full monitoring stack
./docker-deploy.sh -m multi
```

## 📋 **Available Deployment Options**

### **Single API Mode**
- **Main API**: `./docker-deploy.sh -a main -p 8000`
- **Generic API**: `./docker-deploy.sh -a generic -p 8001`
- **Background**: `./docker-deploy.sh -d -a main`

### **Multi-Service Mode**
- **Full Stack**: `./docker-deploy.sh -m multi`
- **Background**: `./docker-deploy.sh -m multi -d`

## 🛠 **Docker Commands**

### **Build Image**
```bash
# Build with cache
docker build -t anomaly_detection_project:latest .

# Build without cache (force rebuild)
docker build --no-cache -t anomaly_detection_project:latest .
```

### **Run Single API**
```bash
# Main API
docker run -d \
  --name anomaly-main-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/temp:/app/temp \
  anomaly_detection_project:latest \
  uvicorn main:app --host 0.0.0.0 --port 8000

# Generic API
docker run -d \
  --name anomaly-generic-api \
  -p 8001:8001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/temp:/app/temp \
  anomaly_detection_project:latest \
  uvicorn generic_api:app --host 0.0.0.0 --port 8001
```

### **Run Multi-Service Stack**
```bash
# Start all services
docker-compose -f docker-compose.multi.yml up -d

# View logs
docker-compose -f docker-compose.multi.yml logs -f

# Stop all services
docker-compose -f docker-compose.multi.yml down
```

## 🌐 **Service Endpoints**

### **APIs**
| Service | URL | Description |
|---------|-----|-------------|
| **Main API** | http://localhost:8000 | Original API with generic detector |
| **Generic API** | http://localhost:8001 | Dedicated generic anomaly detection API |
| **Nginx** | http://localhost:80 | Load balancer and reverse proxy |

### **Monitoring**
| Service | URL | Credentials |
|---------|-----|------------|
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin/admin |

### **Data Services**
| Service | Host:Port | Description |
|---------|-----------|-------------|
| **Redis** | localhost:6379 | Caching layer |
| **PostgreSQL** | localhost:5432 | Model storage |

## 🔧 **Configuration**

### **Environment Variables**
```bash
# API Configuration
export LOG_LEVEL=INFO
export PYTHONPATH=/app

# Database Configuration
export POSTGRES_DB=anomaly_detection
export POSTGRES_USER=anomaly_user
export POSTGRES_PASSWORD=anomaly_password
```

### **Volume Mounts**
```bash
# Models storage
-v $(pwd)/models:/app/models

# Temporary files
-v $(pwd)/temp:/app/temp

# Monitoring data
-v grafana_data:/var/lib/grafana
-v postgres_data:/var/lib/postgresql/data
```

## 📊 **Monitoring & Health Checks**

### **Health Check Endpoints**
```bash
# Main API health
curl http://localhost:8000/health

# Generic API health
curl http://localhost:8001/health

# Nginx health
curl http://localhost:80/health
```

### **Container Health Checks**
```bash
# Check container status
docker ps

# View container logs
docker logs anomaly-main-api
docker logs anomaly-generic-api

# Check health status
docker inspect anomaly-main-api | grep Health -A 10
```

## 🚀 **Production Deployment**

### **1. AWS ECS**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker tag anomaly_detection_project:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/anomaly-detection:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/anomaly-detection:latest
```

### **2. Google Cloud Run**
```bash
# Build and push to GCR
docker tag anomaly_detection_project:latest gcr.io/PROJECT_ID/anomaly-detection:latest
docker push gcr.io/PROJECT_ID/anomaly-detection:latest

# Deploy to Cloud Run
gcloud run deploy anomaly-detection \
  --image gcr.io/PROJECT_ID/anomaly-detection:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### **3. Azure Container Instances**
```bash
# Build and push to ACR
az acr build --registry myregistry --image anomaly-detection:latest .

# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name anomaly-detection \
  --image myregistry.azurecr.io/anomaly-detection:latest \
  --ports 8000 8001
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **Port Already in Use**
```bash
# Find process using port
lsof -i :8000
lsof -i :8001

# Kill process
kill -9 <PID>

# Or use different port
./docker-deploy.sh -a main -p 8002
```

#### **Permission Denied**
```bash
# Fix script permissions
chmod +x docker-deploy.sh

# Fix volume permissions
sudo chown -R $USER:$USER models temp
```

#### **Container Won't Start**
```bash
# Check logs
docker logs anomaly-main-api

# Check container status
docker ps -a

# Remove and recreate
docker rm anomaly-main-api
./docker-deploy.sh -b -a main
```

#### **Memory Issues**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory
# Or add to docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 2G
```

### **Debug Commands**
```bash
# Enter running container
docker exec -it anomaly-main-api /bin/bash

# View real-time logs
docker logs -f anomaly-main-api

# Check resource usage
docker stats anomaly-main-api

# Inspect container
docker inspect anomaly-main-api
```

## 📈 **Performance Optimization**

### **Resource Limits**
```yaml
# In docker-compose.multi.yml
services:
  main-api:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
```

### **Scaling**
```bash
# Scale API instances
docker-compose -f docker-compose.multi.yml up --scale main-api=3 --scale generic-api=2
```

### **Caching**
```bash
# Enable Redis caching
# Add to API environment:
REDIS_URL=redis://redis:6379
CACHE_TTL=3600
```

## 🔒 **Security**

### **Production Security**
```bash
# Use secrets for sensitive data
echo "your-secret-key" | docker secret create api_key -

# Run with non-root user
docker run --user 1000:1000 anomaly_detection_project:latest

# Use read-only filesystem
docker run --read-only anomaly_detection_project:latest
```

### **Network Security**
```yaml
# In docker-compose.multi.yml
networks:
  anomaly_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 📝 **Maintenance**

### **Backup**
```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Backup database
docker exec anomaly-postgres pg_dump -U anomaly_user anomaly_detection > backup.sql
```

### **Updates**
```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
./docker-deploy.sh -b -m multi

# Rolling update
docker-compose -f docker-compose.multi.yml up -d --no-deps main-api
```

### **Cleanup**
```bash
# Stop and remove all containers
./docker-deploy.sh -s

# Clean up everything
./docker-deploy.sh -c

# Remove unused images
docker image prune -a
```

## ✅ **Verification**

### **Test Deployment**
```bash
# Test Main API
curl -X GET http://localhost:8000/health
curl -X GET http://localhost:8000/

# Test Generic API
curl -X GET http://localhost:8001/health
curl -X GET http://localhost:8001/algorithms

# Test with sample data
python3 test_deployment.py
```

### **Load Testing**
```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test API performance
ab -n 1000 -c 10 http://localhost:8000/health
ab -n 1000 -c 10 http://localhost:8001/health
```

## 🎯 **Ready for Production!**

Your anomaly detection APIs are now fully containerized and ready for deployment across any environment! 🚀
