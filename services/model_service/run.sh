echo "Starting MLflow stack: Mlflow, MinIO, PostGres"
docker compose --env-file .env.dev up -d