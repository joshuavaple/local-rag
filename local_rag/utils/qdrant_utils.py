from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from .logger import get_logger


logger = get_logger(__name__)

def get_qdrant_distance_metric(metric_name: str) -> Distance:
    """Convert string distance metric to Qdrant Distance enum
    
    Args:
        metric_name (str): Distance metric name (COSINE, DOT, EUCLIDEAN, MANHATTAN)
    
    Returns:
        Distance: Qdrant Distance enum value
    
    Raises:
        ValueError: If metric_name is not supported
    """
    distance_mapping = {
        "COSINE": Distance.COSINE,
        "DOT": Distance.DOT,
        "EUCLID": Distance.EUCLID,
        "MANHATTAN": Distance.MANHATTAN
    }
    
    metric_name = metric_name.upper()
    if metric_name not in distance_mapping:
        supported_metrics = ", ".join(distance_mapping.keys())
        raise ValueError(f"Unsupported distance metric: {metric_name}. Supported metrics: {supported_metrics}")
    
    return distance_mapping[metric_name]

def setup_qdrant_client_and_collection(cfg: dict) -> QdrantClient:
    """Initiate the Qdrant client with service URL, and a collection with collection name and vector size. 
    If collection doesn't exist, create it.
    """

    qdrant_url = cfg["embedding"]["qdrant_svc_url"]
    collection_name = cfg["embedding"]["qdrant_collection_name"]
    vector_size = cfg["embedding"]["vector_size"]
    distance_metric_name = cfg["embedding"]["distance_metric"]
    distance_metric = get_qdrant_distance_metric(distance_metric_name)
    
    client = QdrantClient(url=qdrant_url)
    
    # Create the collection if it doesn't exist
    if client.collection_exists(collection_name=collection_name):
        logger.info(f"Collection {collection_name} already exists. It will be used.")
    else:
        logger.warning(f"Collection {collection_name} does not exist. Creating with specifications...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance_metric),
        )
        logger.info("Created new collection.")
    
    return client