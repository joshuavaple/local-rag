import yaml
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from app.utils.logger import get_logger
from app.utils.mlflow_utils import log_deployment_ready_model, check_existing_experiment
import mlflow
from mlflow import MlflowClient

logger = get_logger(__name__)

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

# IMPORTANT: when using __file__, it is w.r.t this script
# thus:
# Path(__file__).resolve().parents[0] -> ./services/embedding_service/app/models/ (current directory)
# Path(__file__).resolve().parents[1] -> ./services/embedding_service/app/
# Path(__file__).resolve().parents[2] -> ./services/embedding_service/ (CORRECT)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yml"
CACHE_DIR = Path(__file__).resolve().parents[2] / "model_cache"


logger.debug(f"Device in use: {DEVICE}")

class EmbeddingModel:
    """Wrapper for embedding model (with local caching)."""

    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = DEVICE

    def load_model(self):
        """Load the model from MLflow registry."""
        if self.model is not None:
            return  # Already loaded
            
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)

        model_name = cfg["model"]["name"]
        alias = cfg["model"]["alias"]
        experiment_name = cfg["model"]["experiment_name"]
        mlflow_tracking_uri = cfg["model"]["mlflow_tracking_uri"]

        mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
        client = MlflowClient()
        model_uri = f"models:/{model_name}@{alias}"

        try:
            logger.debug("Loading registered model from MLflow registry...")
            model = mlflow.sentence_transformers.load_model(model_uri)
            logger.debug("Embedding model loaded successfully.")
        except Exception as e:
            logger.warning(e)
            logger.warning(
                f"Embeding model URI {model_uri} not found. Download, log and register..."
            )
            check_existing_experiment(experiment_name=experiment_name)
            model_info = log_deployment_ready_model(
                experiment_name=experiment_name, model_name=model_name
            )

            mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
            latest_version = client.get_registered_model(
                name=model_name
            ).latest_versions[0]
            client.set_registered_model_alias(
                name=model_name, alias=alias, version=latest_version.version
            )
            model = mlflow.sentence_transformers.load_model(model_uri)

        self.model = model
        self.model_name = model_name

    def encode(self, texts):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )
        return embeddings.tolist()

# Global instance
embedding_model = EmbeddingModel()

class EmbeddingModelOld:
    """Singleton wrapper for embedding model (with local caching)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f)

            model_name = cfg["model"]["name"]
            alias = cfg["model"]["alias"]
            experiment_name = cfg["model"]["experiment_name"]
            mlflow_tracking_uri = cfg["model"]["mlflow_tracking_uri"]

            mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
            client = MlflowClient()
            model_uri = f"models:/{model_name}@{alias}"

            try:
                logger.debug("Loading registered model from MLflow registry...")
                model = mlflow.sentence_transformers.load_model(model_uri)
                logger.debug("Embedding model loaded successfully.")
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    f"Embeding model URI {model_uri} not found. Download, log and register..."
                )
                model_info = log_deployment_ready_model(
                    experiment_name=experiment_name, model_name=model_name, alias=alias
                )

                mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
                latest_version = client.get_registered_model(
                    name=model_name
                ).latest_versions[0]
                client.set_registered_model_alias(
                    name=model_name, alias=alias, version=latest_version.version
                )

            cls._instance = super().__new__(cls)
            cls._instance.model = model
            cls._instance.model_name = model_name
            cls._instance.device = DEVICE

        return cls._instance

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )
        return embeddings.tolist()
