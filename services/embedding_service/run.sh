#!/bin/bash
echo "Starting local embedding service..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
