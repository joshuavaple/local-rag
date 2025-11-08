## Setup Environment
1. Create te conda environment with the `conda.yml` file and activate it
2. Run the `./create_docker_volumes.sh` to create Docker volumes for some of the services (check individual service Docker compose files for details).

## Start Up Services
1. `model_service`
    - a MLflow compose project to log and register the pretrained embedding model (all-MiniLM-L12-v2) in the `embedding_service`
    - execute the `model_service/run.sh` script to start it.

2. `embedding_service`
    - A FastAPI app to serve the embedding model from an endpoint
    - execute the `embedding_service/run.sh` script to start it.
    - Under the hood, the `main.py` file does the following:
        - Checks if the model of the specified URI (name and alias) exists in the MLflow registry. If not, download it (from huggingface), log and register it.
        - Preloads the model from the registry with lifespan
        - Exposes a FastAPI endpoint to embed input text with the schema `{"texts":[text1, text2]}` (see the relevant pydantic model)
3. `embedding_service_mlflow`
    - Similar function as #2 above but using `mlflow serve` CLI.
    - 2 methods:
        - `serve_from_registry.sh`: pull model from registry and deploy a FastAPI server everytime, expose via host port 5001.
        - Build image and run the embedding model via a container with compose, via host port 5002:
            - `generate_dockerfile.sh`: points to the mocal mlflow tracking URI and generate a localizes directory to store the mlflow model, then an associated dockerfile to build an image.
            - `build_image.sh`: based on the dockerfile above, build the service image to local Docker registry.
            - `docker-compose.yml`: compose file to up the server based on the image above.
        `run.sh` and `docker-compose.yml`: serve from a pre-created image preloaded with the embedding model. The image is created separately.

3. `vectordb_service`
    - A Qdrant vector database compose project to store embeddings
    - Execute the `vectordb_service/run.sh` script to start it.

## Pipelines
1. `embedding_pipeline`
    - `embed_corpus.py`: take the whole copus, chunk all articles by number of tokens, followed by embedding the chunks with the `embedding_service` model endpoint above and upserting in batches to the vector database collection. Check the `config.yml` for relevant parameters like chunk size and overlap.
    - `data/`: contains the whole copus of the benchmark `MultiHop RAG`

## Localhost Port List
- 8000: embedding
- 5000: mlflow server
- 5001: mlflow embedding model server (no container, model loaded everytime)
- 5002: mlflow embedding model server (in container, model preloaded)
- 5432: postgres
- 9000: minio
- 6333: qdrant