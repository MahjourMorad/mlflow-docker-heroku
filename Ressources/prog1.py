import mlflow
from mlflow import log_metric, log_param, log_artifacts, set_tracking_uri
if __name__ == "__main__":
    mlflow.set_tracking_uri('https://manip.herokuapp.com')
    mlflow.set_experiment(experiment_name='Demo')
    with mlflow.start_run(run_name='FirstRun'):
        mlflow.log_param('b',2)
        for a in range(5):
            mlflow.log_metric('a',a)


        
            
        
            
