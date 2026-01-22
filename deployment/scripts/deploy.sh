#!/bin/bash

# Deployment script for Legal Contract Chatbot

set -e

echo "Deploying Legal Contract Chatbot..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Load environment variables
source .env

# Build Docker image
echo "Building Docker image..."
docker build -t contract-chatbot:latest -f docker/Dockerfile .

# Check if docker-compose is being used
if [ "$1" == "compose" ]; then
    echo "Starting with docker-compose..."
    docker-compose -f docker/docker-compose.yml up -d
    echo "Deployment complete! Access the app at http://localhost:8501"
else
    echo "Running Docker container..."
    docker run -d \
        --name contract-chatbot \
        -p 8501:8501 \
        --env-file .env \
        -v $(pwd)/chroma_db:/app/chroma_db \
        -v $(pwd)/RFP_parsed.json:/app/RFP_parsed.json \
        contract-chatbot:latest
    
    echo "Deployment complete! Access the app at http://localhost:8501"
fi
