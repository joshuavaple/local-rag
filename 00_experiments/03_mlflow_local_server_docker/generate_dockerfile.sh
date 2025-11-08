export MLFLOW_TRACKING_URI=http://localhost:5000/
mlflow models generate-dockerfile --model-uri models:/all-MiniLM-L12-v2@champion --env-manager conda