from fastapi import APIRouter
from pydantic import BaseModel
from app.models.embedding_model import EmbeddingModel

router = APIRouter()

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]

@router.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    embedder = EmbeddingModel()
    vectors = embedder.encode(request.texts)
    return EmbedResponse(model=embedder.model_name, embeddings=vectors)

@router.get("/health")
async def health():
    return {"status": "ok"}
