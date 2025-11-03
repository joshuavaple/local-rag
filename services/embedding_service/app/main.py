from fastapi import FastAPI
# from app.routers import embed
from app.routers import mock # mock route to test
from app.utils.logger import get_logger

from app.models.embedding_model_mlflow import embedding_model
from contextlib import asynccontextmanager
from pydantic import BaseModel


class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    # model: str
    embeddings: list[list[float]]

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Embedding service starting...")
    logger.info("Loading embedding model...")
    embedding_model.load_model()
    logger.info("Embedding model loaded successfully!")
    yield
    # Shutdown
    logger.info("Cleaning up resources and shutting down...")

# the parameters will show in the swagger UI
app = FastAPI(
    title="Local Embedding Service",
    description="Embed text locally using SentenceTransformers",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    embeddings = embedding_model.encode(request.texts)
    return EmbedResponse(embeddings=embeddings)


# other routers
# app.include_router(embed.router)
app.include_router(mock.router)

@app.get("/")
def root():
    return {"message": "Embedding service is up. Use POST /embed"}
