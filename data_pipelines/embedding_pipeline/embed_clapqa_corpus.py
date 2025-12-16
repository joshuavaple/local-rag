from qdrant_client import models
from pathlib import Path
import yaml
import argparse
from datasets import load_dataset
from local_rag.utils.logger import get_logger
from local_rag.utils.mlflow_utils import get_mlflow_embeddings
from local_rag.utils.qdrant_utils import setup_qdrant_client_and_collection
from local_rag.core.text_cleaning import clean_text
from local_rag.core.chunking import SentenceTextSplitter
import time


logger = get_logger(__name__)



def load_config(config_path:str, **overrides:dict) -> dict:
    """Load configuration from YAML file with optional CLI overrides
    
    Args:
        config_path (str): Path to the YAML configuration file (required)
        **overrides: CLI argument overrides in dot notation format
    
    Returns:
        dict: Loaded configuration with overrides applied
    
    Raises:
        ValueError: If config_path is None or empty
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None or config_path == "":
        raise ValueError("config_path is required and cannot be None or empty")
    
    if config_path.endswith(".yml") is False and config_path.endswith(".yaml") is False:
        raise ValueError("config_path must point to a YAML file with .yml or .yaml extension")
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Apply CLI overrides to nested config
    for key, value in overrides.items():
        if value is not None:  # Only override if value was provided
            keys = key.split('.')
            current = cfg
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
    
    logger.debug(f"Loaded config from {config_path}: {cfg}")
    return cfg


def parse_args():
    """Parse command line arguments for runtime overrides"""
    parser = argparse.ArgumentParser(description="Embed CLAP-QA corpus into vector database")
    
    # Config file
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    
    # Service URLs - most likely to be overridden in different environments
    parser.add_argument("--qdrant-url", type=str, 
                       help="Qdrant service URL (overrides config)")
    parser.add_argument("--embedding-url", type=str, 
                       help="Embedding service URL (overrides config)")
    
    # Collection settings
    parser.add_argument("--collection-name", type=str, 
                       help="Qdrant collection name (overrides config)")
    parser.add_argument("--vector-size", type=int, 
                       help="Vector size (overrides config)")
    parser.add_argument("--distance-metric", type=str, 
                       choices=["COSINE", "DOT", "EUCLIDEAN", "MANHATTAN"],
                       help="Distance metric for vector similarity (overrides config)")
        
    # Processing parameters
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--max-docs", type=int, 
                       help="Maximum number of documents to process (for testing)")
    
    # Dataset parameters
    parser.add_argument("--dataset-name", type=str,
                       help="HuggingFace dataset name (overrides config)")
    parser.add_argument("--dataset-split", type=str,
                       help="Dataset split (overrides config)")
       
    return parser.parse_args()








def main():
    """Main pipeline function"""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration with CLI overrides
    cfg = load_config(
        config_path=args.config,
        **{
            'embedding.qdrant_svc_url': args.qdrant_url,
            'embedding.embedding_svc_url': args.embedding_url,
            'embedding.qdrant_collection_name': args.collection_name,
            'embedding.vector_size': args.vector_size,
            'embedding.distance_metric': args.distance_metric,
            'data.dataset_name': args.dataset_name,
            'data.split': args.dataset_split,
        }
    )
    
    logger.info("Starting CLAP-QA corpus embedding pipeline")
    
    # Setup Qdrant client and collection
    client = setup_qdrant_client_and_collection(cfg)
    
    # Load and preprocess data
    dataset_name = cfg["data"]["dataset_name"]
    dataset_split = cfg["data"]["split"]
    
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=dataset_split)
    df = dataset.to_pandas()
    
    # Limit documents for testing if specified
    if args.max_docs:
        logger.info(f"Limiting to {args.max_docs} documents for testing")
        df = df.head(args.max_docs)
    
    # prepare document-level texts
    logger.info("Preparing document-level texts")
    df["doc_id"] = df["id"].str.split("_").str[0]
    df_doc = df.groupby("doc_id", as_index=False).agg({"text": " ".join, "title": "first"})
    
    # clean texts and prepare lists of texts and titles (metadatas)
    logger.info("Cleaning texts")
    df_doc["cleaned_text"] = df_doc["text"].apply(lambda x: clean_text(x))
    cleaned_texts = df_doc["cleaned_text"].tolist()
    titles = df_doc["title"].tolist()
    
    # split texts into sentences
    logger.info("Splitting texts into sentences")
    sentence_splitter = SentenceTextSplitter(keep_separator="end")
    documents = []
    for text, title in zip(cleaned_texts, titles):
        docs = sentence_splitter.create_documents(
            texts=[text],
            metadatas=[{"title": title}]
        )
        documents.extend(docs)
    
    logger.info(f"Total documents after splitting: {len(documents)}")
    
    # Configuration for processing
    collection_name = cfg["embedding"]["qdrant_collection_name"]
    embedding_url = cfg["embedding"]["embedding_svc_url"]
    batch_size = args.batch_size
    
    # Embed and store documents
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        texts = [doc.page_content for doc in batch]
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        # Get embeddings
        embeddings = get_mlflow_embeddings(texts, embedding_url)
        
        # Prepare points for Qdrant
        points = []
        for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
            point_id = i + j
            points.append(models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": doc.page_content,
                    "title": doc.metadata.get("title", ""),
                    "doc_id": point_id
                }
            ))
        
        # Upload to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        # Small delay to avoid overwhelming the services
        time.sleep(0.1)
    
    logger.info(f"Successfully embedded and stored {len(documents)} documents in collection '{collection_name}'")


if __name__ == "__main__":
    main()