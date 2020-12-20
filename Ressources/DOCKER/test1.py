import os
import mlflow
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    mlflow.set_experiment(experiment_name='blablabal')
    with mlflow.start_run(run_name="test run name"):
        mlflow.log_param('b',2)
        for a in range(10):
            mlflow.log_metric('a',a)
