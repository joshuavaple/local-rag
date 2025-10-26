#!/bin/bash
set -e # exit immediately on error

VOLUMES=(
    "rag-mlflow-postgres-metadata"
    "rag-mlflow-minio-artifacts"
    "rag-qdrant-storage"
)

echo "Creating required Docker volumes for services..."

for VOLUME in "${VOLUMES[@]}"; do
    if docker volume inspect "$VOLUME" >/dev/null 2>&1; then
        echo "✅ Volume '$VOLUME' already exists — skipping."
    else
        docker volume create "$VOLUME"
        echo "📦 Created volume: $VOLUME"
    fi
done

echo "All volumes are ready. List of all volumes below:"
docker volume ls