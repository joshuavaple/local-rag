import sys
sys.path.append("../../.")

import yaml
from pathlib import Path
import mlflow
from mlflow import MlflowClient
from src.utils.logger import get_logger
from src.utils.mlflow_utils import log_deployment_ready_model, check_existing_experiment
from sentence_transformers import SentenceTransformer
from mlflow.models.signature import infer_signature



logger = get_logger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[0] / "config.yml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

model_name = cfg["model"]["name"]
alias = cfg["model"]["alias"]
experiment_name = cfg["model"]["experiment_name"]
mlflow_tracking_uri = cfg["model"]["mlflow_tracking_uri"]

mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
client = MlflowClient()
model_uri = f"models:/{model_name}@{alias}"

def main():
    logger.info("Registering embedding model to MLflow registry...")
    # Ensure experiment exists, restore if soft-deleted
    check_existing_experiment(experiment_name=experiment_name)

    # Download and log the SentenceTransformer model
    model_info = log_deployment_ready_model(
        experiment_name=experiment_name, model_name=model_name
    )

    # register the model to the MLflow registry
    mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
    latest_version = client.get_registered_model(
        name=model_name
    ).latest_versions[0]

    # Set alias to the latest version
    client.set_registered_model_alias(
        name=model_name, alias=alias, version=latest_version.version
    )
    logger.info(
        f"Model {model_name} registered successfully with alias '{alias}' pointing to version {latest_version.version}."
    )

if __name__ == "__main__":
    main()