import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/anastasiarblv/mlops_project_github.mlflow")

dagshub.init(repo_owner='anastasiarblv', repo_name='mlops_project_github', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)