import mlflow
from sentence_transformers import SentenceTransformer
from mlflow.models.signature import infer_signature
from .logger import get_logger


logger = get_logger(__name__)

def log_deployment_ready_model(model_name:str, alias:str, experiment_name: str = None):
    """Create a production-ready semantic search model."""
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        logger.debug(f"Started run with info: {run.info}")

        model = SentenceTransformer(model_name)

        sample_input = ["input text"]
        sample_output = model.encode(sample_input)
        signature = infer_signature(model_input=sample_input, model_output=sample_output)

        model_info = mlflow.sentence_transformers.log_model(
            model=model,
            name=model_name,
            signature=signature,
        )

        logger.debug(f"Logged model URI: {model_info.model_uri}")
        return model_info
