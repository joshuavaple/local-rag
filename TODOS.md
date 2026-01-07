[x] Build the embedding service with FastAPI (non-container)
[x] Set up the Qdrant service with Docker volume
[x] Set up the MLflow serice with Docker volume
[x] Improve the embedding service modularity by separating the model download part and storing the model in a MLflow server
[x] Experiment chunking and embedding
[ ] Move the corpus file to a public S3 to avoid committing data file to git
[x] Containerize the embedding service (with mlflow)
[ ] Consolidate all model artifacts in a common folder - including embedding model and generative model
[ ] Optimize the mlflow embedding service image size
[ ] Change the non-mlflow embedding service endpoint I/O schemas to be consistent with mlflow ({"inputs": [query_text]}, {"predictions": [embeddings]})
[ ] How to have a batch/async embedding service - currently handles one request at a time.
