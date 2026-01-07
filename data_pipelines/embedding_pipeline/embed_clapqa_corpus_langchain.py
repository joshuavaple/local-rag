from qdrant_client import models
import argparse
from datasets import load_dataset
from local_rag.utils.logger import get_logger
from local_rag.utils.config_utils import load_config
from local_rag.utils.mlflow_utils import get_mlflow_embeddings
from local_rag.utils.qdrant_utils import setup_qdrant_client_and_collection
from local_rag.core.text_cleaning import clean_text
from local_rag.core.chunking import SentenceTextSplitter, generate_chunk_id
import time
from langchain_qdrant import QdrantVectorStore
from local_rag.core.embeddings import CustomEmbeddings


logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments for runtime overrides"""
    parser = argparse.ArgumentParser(description="Embed CLAP-QA corpus into vector database")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    
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
                       choices=["COSINE", "DOT", "EUCLID", "MANHATTAN"],
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
    # I. CONFIGURATION AND SETUP
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

    # Configuration for processing
    collection_name = cfg["embedding"]["qdrant_collection_name"]
    embedding_url = cfg["embedding"]["embedding_svc_url"]
    qdrant_svc_url = cfg["embedding"]["qdrant_svc_url"]

    # Other runtime parameters
    batch_size = args.batch_size
    max_docs = args.max_docs

    # Configure vector_store using langchain interface:
    embeddings = CustomEmbeddings(endpoint_url=embedding_url)
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=qdrant_svc_url,
    )


    # II. DATA LOADING AND PREPROCESSING
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=dataset_split)
    df = dataset.to_pandas()
    
    # Limit documents for testing if specified
    if max_docs:
        logger.info(f"Limiting to {max_docs} documents for testing")
        df = df.head(max_docs)
    
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
    # note: LangChain uses "Document" to refer to chunk and its metadata, while elsewhere we use "document" to refer to the original full text
    
    
    # Create documents and uuids
    documents = []
    for text, title in zip(cleaned_texts, titles):
        docs = sentence_splitter.create_documents(
            texts=[text],
            metadatas=[{"title": title}]
        )
        documents.extend(docs)
    uuids = [generate_chunk_id(chunk_text=document.page_content) for document in documents]
    logger.info(f"Total documents after splitting: {len(documents)}")

    # III. EMBEDDING AND STORAGE BY BATCHES
    for i in range(0, len(documents), batch_size):
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        # get the batch documents and their respective uuids
        batch_documents = documents[i:i+batch_size]
        batch_uuids = uuids[i:i+batch_size]
        
        # Add documents to Qdrant via LangChain interface   
        vector_store.add_documents(documents=batch_documents, ids=batch_uuids)
        
        # Small delay to avoid overwhelming the services
        time.sleep(0.1)
    
    logger.info(f"Successfully embedded and stored {len(documents)} documents in collection '{collection_name}'")


if __name__ == "__main__":
    main()