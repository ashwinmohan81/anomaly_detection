#!/bin/bash

# Docker Deployment Script for Anomaly Detection APIs
# Supports both single API and multi-service deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="single"
PORT="8000"
BUILD_ARGS=""

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

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Anomaly Detection Docker Deploy${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE        Deployment mode: single, multi (default: single)"
    echo "  -p, --port PORT        Port to expose (default: 8000)"
    echo "  -b, --build            Force rebuild image"
    echo "  -d, --detach           Run in background"
    echo "  -s, --stop             Stop running containers"
    echo "  -c, --clean            Clean up containers and images"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run API on port 8000"
    echo "  $0 -p 8001                           # Run API on port 8001"
    echo "  $0 -m multi                          # Run with full monitoring stack"
    echo "  $0 -b                                # Rebuild and run API"
    echo "  $0 -s                                # Stop all containers"
    echo "  $0 -c                                # Clean up everything"
}

# Parse command line arguments
DETACH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -b|--build)
            BUILD_ARGS="--no-cache"
            shift
            ;;
        -d|--detach)
            DETACH="-d"
            shift
            ;;
        -s|--stop)
            print_header
            print_status "Stopping all anomaly detection containers..."
            docker-compose -f docker-compose.yml down 2>/dev/null || true
            docker-compose -f docker-compose.multi.yml down 2>/dev/null || true
            docker stop anomaly-detection-api 2>/dev/null || true
            print_status "All containers stopped"
            exit 0
            ;;
        -c|--clean)
            print_header
            print_status "Cleaning up containers and images..."
            docker-compose -f docker-compose.yml down --volumes --remove-orphans 2>/dev/null || true
            docker-compose -f docker-compose.multi.yml down --volumes --remove-orphans 2>/dev/null || true
            docker stop anomaly-detection-api 2>/dev/null || true
            docker rm anomaly-detection-api 2>/dev/null || true
            docker rmi anomaly_detection_project:latest 2>/dev/null || true
            print_status "Cleanup completed"
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

print_header

# Validate inputs
if [[ "$MODE" != "single" && "$MODE" != "multi" ]]; then
    print_error "Invalid mode: $MODE. Use 'single' or 'multi'"
    exit 1
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p models temp monitoring/grafana/dashboards

# Build Docker image
print_status "Building Docker image..."
docker build $BUILD_ARGS -t anomaly_detection_project:latest .

if [[ $? -ne 0 ]]; then
    print_error "Docker build failed"
    exit 1
fi

print_status "Docker image built successfully"

# Deploy based on mode
if [[ "$MODE" == "single" ]]; then
    print_status "Deploying single API mode..."
    
    # Stop any existing containers
    docker stop anomaly-detection-api 2>/dev/null || true
    docker rm anomaly-detection-api 2>/dev/null || true
    
    # Run container
    print_status "Starting anomaly detection API on port $PORT..."
    docker run $DETACH \
        --name anomaly-detection-api \
        -p $PORT:8000 \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/temp:/app/temp \
        anomaly_detection_project:latest
    
    if [[ $? -eq 0 ]]; then
        print_status "Anomaly detection API is running on http://localhost:$PORT"
        print_status "Health check: http://localhost:$PORT/health"
        print_status "API documentation: http://localhost:$PORT/docs"
    else
        print_error "Failed to start anomaly detection API"
        exit 1
    fi

else
    print_status "Deploying multi-service mode..."
    
    # Stop any existing containers
    docker-compose -f docker-compose.multi.yml down 2>/dev/null || true
    
    # Start all services
    print_status "Starting all services..."
    docker-compose -f docker-compose.multi.yml up $DETACH
    
    if [[ $? -eq 0 ]]; then
        print_status "All services are running:"
        print_status "  Anomaly Detection API: http://localhost:8000"
        print_status "  Nginx Load Balancer: http://localhost:80"
        print_status "  Prometheus: http://localhost:9090"
        print_status "  Grafana: http://localhost:3000 (admin/admin)"
        print_status "  Redis: localhost:6379"
        print_status "  PostgreSQL: localhost:5432"
    else
        print_error "Failed to start services"
        exit 1
    fi
fi

print_status "Deployment completed successfully!"
