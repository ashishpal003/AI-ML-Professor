import mlflow

class MLFlowTracker:

    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name=experiment_name)

    def __enter__(self):
        self.run = mlflow.start_run()
        return self
    
    def log_params(self, params: dict):
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
    
    def log_artifact(self, path: str):
        mlflow.log_artifact(path)
    
    def __exit__(self, exc_type, exc, tb):
        mlflow.end_run()