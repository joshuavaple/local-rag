import numpy as np
from sklearn.linear_model import LogisticRegression
import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
from mlflow.models import infer_signature


mlflow.set_tracking_uri("http://localhost:5000")
model_name = "test_logistic_regression_model"
client = MlflowClient()
if __name__ == "__main__":
    with mlflow.start_run() as run:
        X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        lr = LogisticRegression()
        lr.fit(X, y)
        score = lr.score(X, y)
        print(f"Model score: {score}")
        mlflow.log_metric("score", score)
        predictions = lr.predict(X)
        signature = infer_signature(X, predictions)
        model_info = mlflow.sklearn.log_model(lr, "model", signature=signature)
        print(f"Model logged in run {run.info.run_id}")

        mlflow.register_model(model_uri=model_info.model_uri, name=model_name)

        latest_version = client.get_registered_model(
            name=model_name
        ).latest_versions[0]
        alias = "test_latest"
        # Set alias to the latest version
        client.set_registered_model_alias(
            name=model_name, alias=alias, version=latest_version.version
        )

        print(
            f"Model {model_name} registered successfully with alias '{alias}'."
        )