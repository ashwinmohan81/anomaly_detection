# 🚀 Seamless Model Hosting Guide

## Quick Start (5 minutes)

### 1. Local Development
```bash
# Start the API
./deploy.sh development docker

# Test the deployment
python3 test_deployment.py

# View logs
docker-compose logs -f anomaly-api
```

### 2. Production Deployment
```bash
# Deploy to production
./deploy.sh production docker

# Access the API
curl http://localhost:8000/health
```

## 🌐 Deployment Options

### Option 1: Docker (Recommended for Start)
**Best for**: Quick deployment, development, small teams
**Cost**: $5-20/month
**Setup time**: 5 minutes

```bash
# Deploy with Docker Compose
./deploy.sh production docker

# Access services:
# - API: http://localhost:8000
# - Grafana: http://localhost:3000 (admin/admin123)
# - Prometheus: http://localhost:9090
```

### Option 2: Cloud Platforms
**Best for**: Production, scalability, managed services
**Cost**: $20-100/month
**Setup time**: 15-30 minutes

#### AWS ECS
```bash
# Deploy to AWS
./deploy.sh production aws

# Access via AWS Load Balancer
```

#### Google Cloud Run
```bash
# Deploy to GCP
./deploy.sh production gcp

# Access via Cloud Run URL
```

#### Azure Container Instances
```bash
# Deploy to Azure
./deploy.sh production azure

# Access via Azure public IP
```

### Option 3: Kubernetes
**Best for**: Enterprise, high availability, microservices
**Cost**: $50-500/month
**Setup time**: 30-60 minutes

```bash
# Deploy to Kubernetes
./deploy.sh production kubernetes

# Check status
kubectl get pods -l app=anomaly-detection-api
```

### Option 4: Serverless
**Best for**: Variable load, cost optimization
**Cost**: $5-50/month
**Setup time**: 10-20 minutes

```bash
# Deploy to AWS Lambda
serverless deploy

# Deploy to Vercel
vercel --prod
```

## 📊 Monitoring & Observability

### Built-in Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and visualization
- **Jaeger**: Distributed tracing
- **Nginx**: Load balancing and access logs

### Key Metrics
- API response times
- Request rates and errors
- Model prediction accuracy
- System resource usage
- Anomaly detection rates

### Access Monitoring
```bash
# View Grafana dashboards
open http://localhost:3000

# View Prometheus metrics
open http://localhost:9090

# View Jaeger traces
open http://localhost:16686
```

## 🔧 Configuration

### Environment Variables
```bash
# Production settings
ENVIRONMENT=production
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379
DATABASE_URL=postgresql://user:pass@postgres:5432/anomaly_detection

# Security settings
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=api.yourdomain.com
CORS_ORIGINS=https://yourdomain.com
```

### Resource Limits
```yaml
# Docker Compose resource limits
resources:
  limits:
    cpus: '2.0'
    memory: 2G
  reservations:
    cpus: '1.0'
    memory: 1G
```

### Scaling
```bash
# Scale horizontally
docker-compose up -d --scale anomaly-api=3

# Scale vertically (Kubernetes)
kubectl scale deployment anomaly-detection-api --replicas=5
```

## 🛡️ Security Features

### Built-in Security
- **Rate limiting**: 10 requests/second per IP
- **CORS protection**: Configurable origins
- **Security headers**: XSS, CSRF protection
- **Health checks**: Automatic failover
- **SSL/TLS**: HTTPS support

### Authentication (Optional)
```python
# Add API key authentication
@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not validate_api_key(api_key):
        return JSONResponse({"error": "Unauthorized"}, 401)
    return await call_next(request)
```

## 📈 Performance Optimization

### Caching
```python
# Redis caching for models
import redis
redis_client = redis.Redis(host='redis', port=6379, db=0)

# Cache model predictions
@lru_cache(maxsize=1000)
def cached_predict(model_id, data):
    return model.predict(data)
```

### Database Optimization
```sql
-- Index for faster queries
CREATE INDEX idx_predictions_timestamp ON predictions(created_at);
CREATE INDEX idx_predictions_model_id ON predictions(model_id);
```

### Load Balancing
```nginx
# Nginx load balancing
upstream anomaly_api {
    server anomaly-api-1:8000;
    server anomaly-api-2:8000;
    server anomaly-api-3:8000;
}
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to AWS
        run: ./deploy.sh production aws
```

### GitLab CI
```yaml
# .gitlab-ci.yml
deploy:
  stage: deploy
  script:
    - ./deploy.sh production docker
  only:
    - main
```

## 💰 Cost Optimization

### Resource Sizing
| Environment | CPU | Memory | Cost/Month |
|-------------|-----|--------|------------|
| Development | 0.5 | 512MB | $5-10 |
| Production | 2.0 | 2GB | $20-40 |
| Enterprise | 4.0 | 8GB | $80-160 |

### Auto-scaling
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Service Not Starting
```bash
# Check logs
docker-compose logs anomaly-api

# Check health
curl http://localhost:8000/health
```

#### 2. High Memory Usage
```bash
# Check resource usage
docker stats

# Scale down if needed
docker-compose up -d --scale anomaly-api=1
```

#### 3. Database Connection Issues
```bash
# Check database status
docker-compose exec postgres pg_isready

# Check connection string
echo $DATABASE_URL
```

### Debug Commands
```bash
# View all logs
docker-compose logs

# Execute shell in container
docker-compose exec anomaly-api bash

# Check service status
docker-compose ps

# Restart specific service
docker-compose restart anomaly-api
```

## 📚 Advanced Topics

### Multi-Region Deployment
```bash
# Deploy to multiple regions
./deploy.sh production aws --region us-east-1
./deploy.sh production aws --region eu-west-1
```

### Disaster Recovery
```bash
# Backup database
docker-compose exec postgres pg_dump -U api_user anomaly_detection > backup.sql

# Restore database
docker-compose exec -T postgres psql -U api_user anomaly_detection < backup.sql
```

### Custom Domains
```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name api.yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        proxy_pass http://anomaly_api;
    }
}
```

## 🎯 Production Checklist

### Before Going Live
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure SSL/TLS certificates
- [ ] Set up backup strategy
- [ ] Configure log aggregation
- [ ] Set up alerting
- [ ] Load test the API
- [ ] Set up CI/CD pipeline
- [ ] Document API endpoints
- [ ] Set up rate limiting
- [ ] Configure security headers

### Post-Deployment
- [ ] Monitor error rates
- [ ] Check response times
- [ ] Verify health checks
- [ ] Monitor resource usage
- [ ] Set up alerts
- [ ] Test failover scenarios
- [ ] Update documentation
- [ ] Train team on monitoring

## 🆘 Support

### Getting Help
1. **Check logs**: `docker-compose logs`
2. **Run tests**: `python3 test_deployment.py`
3. **Check health**: `curl http://localhost:8000/health`
4. **View metrics**: `curl http://localhost:8000/metrics`

### Common Commands
```bash
# Quick health check
curl -f http://localhost:8000/health || echo "API is down"

# View recent logs
docker-compose logs --tail=100 anomaly-api

# Restart everything
docker-compose down && docker-compose up -d

# Scale up
docker-compose up -d --scale anomaly-api=3
```

This guide provides everything you need to seamlessly host your anomaly detection models at any scale! 🚀
