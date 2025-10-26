from fastapi import FastAPI
from app.routers import embed
from app.routers import mock # mock route to test
from app.utils.logger import get_logger

logger = get_logger(__name__)

# the parameters will show in the swagger UI
app = FastAPI(
    title="Local Embedding Service",
    description="Embed text locally using SentenceTransformers",
    version="1.0.0",
)

app.include_router(embed.router)
app.include_router(mock.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Embedding service starting...")

@app.get("/")
def root():
    return {"message": "Embedding service is up. Use POST /embed"}
