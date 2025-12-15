import requests
from langchain_core.embeddings import Embeddings


class CustomEmbeddings(Embeddings):
    def __init__(
        self,
        endpoint_url: str,
        headers: dict = None,
        **kwargs   
    ):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.headers = headers or {}
        self.session = requests.Session()
        if self.headers:
            self.session.headers.update(self.headers)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of text documents to embed.
            
        Returns:
            List of embeddings (list of floats for each document).
        """
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.
        
        Args:
            text: Query text to embed.
            
        Returns:
            Embedding as list of floats.
        """
        embeddings = self._get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:

        payload = {
            "inputs": texts
        }
        
        response = self.session.post(url=self.endpoint_url, json=payload)
        response.raise_for_status()
        result = response.json()

        # Common response formats:
        # Option 1: Direct list of embeddings
        if isinstance(result, list):
            return result
        
        # Option 2: MLFLOW - Wrapped in predictions key
        if "predictions" in result:
            return result["predictions"]
        
        # Option 3: Other format - adjust as needed
        if "outputs" in result:
            return result["outputs"]

        raise ValueError(f"Unexpected response format: {result}")