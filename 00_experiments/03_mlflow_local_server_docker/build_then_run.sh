cd mlflow-dockerfile
docker build -t embedding_service:latest .
docker run -p 5002:8080 embedding_service:latest