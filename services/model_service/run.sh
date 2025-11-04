echo "Starting MLflow stack: Mlflow, MinIO, PostGres"
# docker compose --env-file .env.dev up -d
cp .env.dev .env
docker compose up -d