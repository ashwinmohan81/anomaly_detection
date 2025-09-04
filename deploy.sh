#!/bin/bash

# Anomaly Detection API Deployment Script
# Usage: ./deploy.sh [environment] [platform]

set -e

ENVIRONMENT=${1:-development}
PLATFORM=${2:-docker}

echo "🚀 Deploying Anomaly Detection API"
echo "Environment: $ENVIRONMENT"
echo "Platform: $PLATFORM"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    case $PLATFORM in
        docker)
            if ! command -v docker &> /dev/null; then
                print_error "Docker is not installed"
                exit 1
            fi
            if ! command -v docker-compose &> /dev/null; then
                print_error "Docker Compose is not installed"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                print_error "kubectl is not installed"
                exit 1
            fi
            ;;
        aws)
            if ! command -v aws &> /dev/null; then
                print_error "AWS CLI is not installed"
                exit 1
            fi
            ;;
        gcp)
            if ! command -v gcloud &> /dev/null; then
                print_error "Google Cloud CLI is not installed"
                exit 1
            fi
            ;;
    esac
    
    print_status "Prerequisites check passed"
}

# Build Docker image
build_docker_image() {
    print_status "Building Docker image..."
    
    # Create .dockerignore if it doesn't exist
    if [ ! -f .dockerignore ]; then
        cat > .dockerignore << EOF
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv
.venv
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
*.pkl
*.csv
*.log
EOF
    fi
    
    # Build image with environment-specific tag
    docker build -t anomaly-detection:$ENVIRONMENT .
    
    print_status "Docker image built successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    print_status "Deploying with Docker Compose..."
    
    # Create environment-specific compose file
    if [ "$ENVIRONMENT" = "production" ]; then
        COMPOSE_FILE="docker-compose.prod.yml"
    else
        COMPOSE_FILE="docker-compose.yml"
    fi
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_error "Compose file $COMPOSE_FILE not found"
        exit 1
    fi
    
    # Stop existing containers
    docker-compose -f $COMPOSE_FILE down
    
    # Start new containers
    docker-compose -f $COMPOSE_FILE up -d
    
    # Wait for health check
    print_status "Waiting for service to be healthy..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Service is healthy and running on http://localhost:8000"
    else
        print_error "Service health check failed"
        docker-compose -f $COMPOSE_FILE logs
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    print_status "Deploying to Kubernetes..."
    
    # Check if kubectl context is set
    if ! kubectl cluster-info &> /dev/null; then
        print_error "No Kubernetes cluster context found"
        exit 1
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace anomaly-detection --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/ -n anomaly-detection
    
    # Wait for deployment
    print_status "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/anomaly-detection-api -n anomaly-detection
    
    # Get service URL
    SERVICE_URL=$(kubectl get service anomaly-detection-service -n anomaly-detection -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$SERVICE_URL" ]; then
        SERVICE_URL="localhost"
    fi
    
    print_status "Service deployed successfully"
    print_status "Access the API at: http://$SERVICE_URL"
}

# Deploy to AWS ECS
deploy_aws() {
    print_status "Deploying to AWS ECS..."
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured"
        exit 1
    fi
    
    # Get AWS account ID and region
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    AWS_REGION=$(aws configure get region)
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names anomaly-detection --region $AWS_REGION &> /dev/null || \
    aws ecr create-repository --repository-name anomaly-detection --region $AWS_REGION
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    
    # Tag and push image
    docker tag anomaly-detection:$ENVIRONMENT $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/anomaly-detection:$ENVIRONMENT
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/anomaly-detection:$ENVIRONMENT
    
    # Update ECS service
    aws ecs update-service --cluster anomaly-detection --service anomaly-detection-api --force-new-deployment --region $AWS_REGION
    
    print_status "Service updated in AWS ECS"
}

# Deploy to Google Cloud Run
deploy_gcp() {
    print_status "Deploying to Google Cloud Run..."
    
    # Check if gcloud is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
        print_error "Google Cloud not authenticated"
        exit 1
    fi
    
    # Get project ID
    PROJECT_ID=$(gcloud config get-value project)
    
    # Build and push to Google Container Registry
    docker tag anomaly-detection:$ENVIRONMENT gcr.io/$PROJECT_ID/anomaly-detection:$ENVIRONMENT
    docker push gcr.io/$PROJECT_ID/anomaly-detection:$ENVIRONMENT
    
    # Deploy to Cloud Run
    gcloud run deploy anomaly-detection \
        --image gcr.io/$PROJECT_ID/anomaly-detection:$ENVIRONMENT \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 10
    
    print_status "Service deployed to Google Cloud Run"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    # Wait for service to be ready
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Health check passed"
    else
        print_error "Health check failed"
        return 1
    fi
    
    # Run API tests if available
    if [ -f "test_api.py" ]; then
        python3 test_api.py
        print_status "API tests passed"
    fi
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    
    case $PLATFORM in
        docker)
            docker-compose down
            ;;
        kubernetes)
            kubectl delete -f k8s/ -n anomaly-detection --ignore-not-found=true
            ;;
    esac
}

# Main deployment logic
main() {
    check_prerequisites
    
    case $PLATFORM in
        docker)
            build_docker_image
            deploy_docker
            run_tests
            ;;
        kubernetes)
            build_docker_image
            deploy_kubernetes
            run_tests
            ;;
        aws)
            build_docker_image
            deploy_aws
            ;;
        gcp)
            build_docker_image
            deploy_gcp
            ;;
        *)
            print_error "Unsupported platform: $PLATFORM"
            print_status "Supported platforms: docker, kubernetes, aws, gcp"
            exit 1
            ;;
    esac
    
    print_status "Deployment completed successfully! 🎉"
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@"
