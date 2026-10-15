import os

# MLflow 3 prints a hint about its own tooling on import; keep the logs about our runs readable.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
