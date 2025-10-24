import yaml
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from app.utils.logger import get_logger
# import structlog


# logger = structlog.get_logger()
logger = get_logger(__name__)

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
# CONFIG_PATH = Path(".").resolve().parents[1] / "config.yml"
# CACHE_DIR = Path(".").resolve().parents[1] / "model_cache"

# IMPORTANT: when using ".", it is w.r.t the shell script that execute the whole program
# i.e., ./services/embedding_service/run.sh
# CONFIG_PATH = Path(".").resolve() / "config.yml"
# CACHE_DIR = Path(".").resolve() / "model_cache"

# IMPORTANT: when using __file__, it is w.r.t this script
# thus:
# Path(__file__).resolve().parents[0] -> ./services/embedding_service/app/models/ (current directory)
# Path(__file__).resolve().parents[1] -> ./services/embedding_service/app/
# Path(__file__).resolve().parents[2] -> ./services/embedding_service/ (CORRECT)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yml"
CACHE_DIR = Path(__file__).resolve().parents[2] / "model_cache"

# logger.debug(f"Using device: {DEVICE}")
# logger.debug(CONFIG_PATH)
# logger.debug(CACHE_DIR)

# print(f"CONFIG path: {CONFIG_PATH}")
# print(f"CACHE_DIR path: {CACHE_DIR}")

logger.debug(f"Device in use: {DEVICE}")

class EmbeddingModel:
    """Singleton wrapper for embedding model (with local caching)."""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f)

            model_name = cfg["model"]["name"]

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            local_model_dir = CACHE_DIR / model_name.replace("/", "_")

            if not local_model_dir.exists():
                logger.debug(f"🔽 Downloading model {model_name} to {local_model_dir}")
                model = SentenceTransformer(model_name, device=DEVICE, cache_folder=str(local_model_dir))
                model.save(str(local_model_dir))
            else:
                logger.debug(f"✅ Loading model from local cache: {local_model_dir}")
                model = SentenceTransformer(str(local_model_dir), device=DEVICE)

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
