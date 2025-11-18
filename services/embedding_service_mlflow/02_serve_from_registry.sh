#!/bin/bash
export MLFLOW_TRACKING_URI=http://localhost:5000/
mlflow models serve -m models:/all-MiniLM-L12-v2@champion -p 5001 --env-manager conda