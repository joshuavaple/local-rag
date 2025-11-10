# not a clean code, fix later
import sys

sys.path.append("../../.")

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from pathlib import Path
import yaml
from src.utils.logger import get_logger
from utils.chunking import chunk_text_with_overlap, generate_chunk_id
import tiktoken
import json
import requests


logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[0] / "config_wixqa.yml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

logger.debug(f"Loaded config: {cfg}")

MAX_TOKENS = cfg["chunking"]["max_tokens"]
CHUNK_OVERLAP = cfg["chunking"]["chunk_overlap"]
LLM = cfg["chunking"]["downstream_llm"]

QDRANT_SVC_URL = cfg["embedding"]["qdrant_svc_url"]
QDRANT_COLLECTION_NAME = cfg["embedding"]["qdrant_collection_name"]
EMBEDDING_SVC_URL = cfg["embedding"]["embedding_svc_url"]
VECTOR_SIZE = cfg["embedding"]["vector_size"]

CORPUS_PATH = cfg["data"]["path"]

client = QdrantClient(url=QDRANT_SVC_URL)

def main():

    # create the collection if not exist
    if client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
        logger.info(
            f"Collection {QDRANT_COLLECTION_NAME} already exists. It will be used."
        )

    else:
        logger.warning(
            f"Collection {QDRANT_COLLECTION_NAME} does not exist. Creating with specifications..."
        )
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created new collection.")
    
    # chunking and embed
    logger.info("Initializing tokenizer...")
    tokenizer = tiktoken.encoding_for_model(LLM)

    logger.info("Loading corpus data...")
    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj)
            
    points_to_insert = []
    excluded_fields = ["contents", "html_content"]
    for doc_idx, doc in enumerate(docs):
        contents = doc.get("contents", "")
        if not contents:
            continue

        chunks = chunk_text_with_overlap(
            tokenizer=tokenizer, text=contents, max_tokens=MAX_TOKENS, overlap=CHUNK_OVERLAP
        )

        payload = {"inputs": chunks} # endpoint accept {"text":[list of str]}
        resp = requests.post(EMBEDDING_SVC_URL, json=payload)
        resp.raise_for_status()
        embeddings = resp.json()["predictions"]  # list of vectors

        for chunk_id, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            # copy the metadata of the doc to the chunk payload
            payload_doc = {k: v for k, v in doc.items() if k not in excluded_fields}

            # add chunk payload info
            payload_doc["chunk_id"] = chunk_id
            payload_doc["num_tokens"] = len(tokenizer.encode(chunk_text))
            payload_doc["contents_chunk"] = chunk_text

            # insert chunks with ID generated from the chunk text and source URL
            points_to_insert.append(
                models.PointStruct(id=generate_chunk_id(doc["url"], chunk_text), vector=emb, payload=payload_doc)
            )
    # bulk upsert
    logger.info("Bulk-upserting in batches...")
    BATCH_SIZE = 500
    for i in range(0, len(points_to_insert), BATCH_SIZE):
        batch = points_to_insert[i:i+BATCH_SIZE]
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch)
    
    logger.info("Completed.")

if __name__ == "__main__":
    main()