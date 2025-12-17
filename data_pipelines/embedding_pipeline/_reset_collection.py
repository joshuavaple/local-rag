from qdrant_client import QdrantClient, models
from local_rag.utils.config_utils import load_config
from local_rag.utils.logger import get_logger
import argparse


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the configuration YAML file"
    )
    parser.add_argument("--qdrant_url", type=str, help="Qdrant service URL")
    parser.add_argument("--collection_name", type=str, help="Qdrant collection name")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(
        config_path=args.config,
        **{
            "embedding.qdrant_svc_url": args.qdrant_url,
            "embedding.qdrant_collection_name": args.collection_name,
        }
    )
    qdrant_url = cfg["embedding"]["qdrant_svc_url"]
    collection_name = cfg["embedding"]["qdrant_collection_name"]
    logger.info("Resetting CLAP-QA Qdrant collection")

    client = QdrantClient(url=qdrant_url)
    # inform user on how many points will be deleted
    collection_info = client.get_collection(collection_name=collection_name)
    num_points = collection_info.points_count
    logger.info(f"Deleting {num_points} points from collection '{collection_name}'")

    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=models.Filter(must=[])),
        wait=True,  # Optional: wait for completion
    )
    logger.info(f"Collection '{collection_name}' has been reset.")

if __name__ == "__main__":
    main()
