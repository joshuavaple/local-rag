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

3. `vectordb_service`
    - A Qdrant vector database compose project to store embeddings
    - Execute the `vectordb_service/run.sh` script to start it.

## Pipelines
1. `embedding_pipeline`
    - `embed_corpus.py`: take the whole copus, chunk all articles by number of tokens, followed by embedding the chunks with the `embedding_service` model endpoint above and upserting in batches to the vector database collection. Check the `config.yml` for relevant parameters like chunk size and overlap.
    - `data/`: contains the whole copus of the benchmark `MultiHop RAG`