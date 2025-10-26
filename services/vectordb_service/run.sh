#!/bin/bash
echo "Starting Qdrant container"
docker compose up -d
echo "Access web UI at http://localhost:6333/dashboard"
