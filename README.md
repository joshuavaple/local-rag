## Setup Environment
1. Create te conda environment with the `conda.yml` file and activate it
2. Run the `./create_docker_volumes.sh` to create Docker volumes for some of the services (check individual service Docker compose files for details).

## Start Up Services
1. `model_service`
    - a MLflow compose project to log and register the pretrained embedding model (all-MiniLM-L12-v2) in the `embedding_service`
    - execute the `model_service/run.sh` script to start it.
    - Reference: https://www.youtube.com/watch?v=d52I300ojm0&list=WL&index=11

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
    - Reference: https://qdrant.tech/documentation/guides/installation/#docker-and-docker-compose

## Pipelines
1. Corpus chunking and embedding
    - Script location: `embedding_pipeline/embed_clapqa_corpus_langchain.py`
    - This is the main script to clean, reconstruct the articles from the sentences, chunk by sentences, and embed these chunks inside the corpus.
    - The script implements the LangChain interface, which greatly reduces the amount of codes required (see `embed_clapqa_corpus.py` for the native Qdrant implementation)
    - Config is stored in the separate yml file.
    - Sample usage:
        - cd to the folder containing this script
        - run the script with appropriate arguments:
            - Embedding full corpus: 
            ```bash
            python embed_clapqa_corpus_langchain.py \
                --config config_clapqa_langchain.yml
            ```
            - Embedding a subset of 100 rows for testing: 
            ```bash
            python embed_clapqa_corpus_langchain.py \
                --config config_clapqa_langchain.yml \
                --max-docs 100
            ```
        - Optionally, you can override config parameters (but not recommended)
            ```bash
            python embed_clapqa_corpus_langchain.py \
                --config config_clapqa_langchain.yml \
                --qdrant-url http://new-qdrant-url:6333 \
                --collection-name new_collection
            ```
2. Reset a collection
    - Script location: `embedding_pipelines/_reset_collection.py`
    - This must be used with the same config file that was used with the embedding script in #1 abobe.
    - Sample usage:
        - cd to the folder containing this script
        - Run the script with the config file corresponding to the collection you want to reset:
        ```bash
        python _reset_collection.py \
            --config config_clapqa_langchain.yml
        ```


## Project Architecture

This project follows a **microservices + shared libraries** pattern with clear separation of concerns:

### Backend Services (`services/`)
Self-contained, independently deployable services that can run in their own containers:

- **Service Isolation**: Each service in `services/`, if built from source, contains its own utilities and dependencies
- **Independent Deployment**: Services can be deployed service-by-service without external project dependencies
- **Docker-Friendly**: Most services are deployed with Docker, so own utility modules are not required.
- **Custom Logic**: Service-specific implementations and configurations remain local

### RAG Pipeline Clients And Experiments (`data_pipelines/`, `00_experiments/`, `clients/`)
Client-side code that coordinates multiple services and implements RAG workflows:

- **Shared Utilities**: Uses the `local_rag` package for common functionality (logging, MLflow utils, Qdrant utils, etc.)
- **Service Orchestration**: Coordinates multiple backend services to implement end-to-end RAG pipelines
- **Experimentation**: Notebooks and scripts for testing, evaluation, and prototyping

### Truly Reusable Packages
For utilities with broad applicability beyond this project:
- **PyPI Publication**: Stable, well-tested utilities meant to be shared everywhere, especially across server and client side, should be published as separate packages
- **Proper Versioning**: Enables semantic versioning and dependency management
- **Cross-Project Reuse**: Other projects and teams can benefit from shared utilities

### Design Rationale
This approach avoids common anti-patterns while maintaining:
- **Clear Boundaries**: Services vs. clients vs. reusable packages
- **Minimal Coupling**: Services don't depend on project-specific shared code
- **Maintainability**: Code duplication is acceptable for service isolation
- **Scalability**: Pattern works for both small and large deployments

## Localhost Port List
- 8000: embedding
- 8001: vllm service
- 5000: mlflow server
- 5001: mlflow embedding model server (no container, model loaded everytime)
- 5002: mlflow embedding model server (in container, model preloaded)
- 5432: mlflow postgres
- 9000: mlflow minio
- 6333: qdrant
- 5433: evidently postgres
- 8080: evidently service
- 3010: LLM chat UI