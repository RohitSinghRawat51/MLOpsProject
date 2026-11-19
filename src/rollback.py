import sys

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "electricity-forecaster"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

if len(sys.argv) != 2:
    sys.exit("usage: python src/rollback.py <version number>")

version = int(sys.argv[1])
old = client.get_model_version_by_alias(MODEL_NAME, "champion").version
client.set_registered_model_alias(MODEL_NAME, "champion", version)
client.set_model_version_tag(MODEL_NAME, version, "rollback", f"restored as champion (replaced v{old})")
print(f"Champion alias moved from v{old} to v{version}.")