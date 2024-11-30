import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dagshub.init(repo_owner='anastasiarblv', repo_name='2811wp_github', mlflow=True)
mlflow.set_experiment("1_experiment__: basic models")  # Name of the experiment in MLflow
mlflow.set_tracking_uri("https://dagshub.com/anastasiarblv/2811wp_github.mlflow")


data = pd.read_csv(r"C:\Users\honor\Desktop\water_potability.csv") # адрес файл с рабочего стоала компа
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():  
            median_value = df[column].median()  
            df[column].fillna(median_value, inplace=True) 
    return df


train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

X_train = train_processed_data.drop(columns=["Potability"], axis=1)
X_test = test_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]
y_test = test_processed_data["Potability"]

RANDOM_STATE = 42
classifiers = {'RandomForestClassifier': RandomForestClassifier(random_state = RANDOM_STATE),
               'LogisticRegression':LogisticRegression(random_state = RANDOM_STATE)}
RandomForestClassifier_space = {}
LogisticRegression_space = {}

params = {'RandomForestClassifier': RandomForestClassifier_space,
          'LogisticRegression': LogisticRegression_space}

models_ToTune = list(classifiers.values())     # [RandomForestClassifier(random_state=42)]
model_Names_ToTune =  list(classifiers.keys()) # ['RandomForestClassifier']
models_params_ToTune = list(params.values())   # [{'n_estimators': [100, 200, 300, 500, 1000], 'max_depth': [None, 4, 5, 6, 10]}]

nabor = list(zip(model_Names_ToTune, models_ToTune, models_params_ToTune))
#[('RandomForestClassifier',
#  RandomForestClassifier(random_state=42),
#  {'n_estimators': [100, 200, 300, 500, 1000], max_depth': [None, 4, 5, 6, 10]})]

# Start a parent MLflow run to track the overall experiment
with mlflow.start_run(run_name="basic: Water_Potability_Models_Experiment"):
    # Iterate over each model in the dictionary
    for cur_model_Name, cur_model, param in nabor:
        # Start a child run within the parent run for each individual model
        #random_search = RandomizedSearchCV(estimator=cur_model, param_distributions=param, verbose=0, cv=10, n_iter=10, scoring ='accuracy', random_state=42, n_jobs=-1)
        with mlflow.start_run(run_name=cur_model_Name, nested=True): 
            # Train the model on the training data
            cur_model.fit(X_train, y_train)
            cur_model_filename = f"basic_{cur_model_Name}.pkl"
            pickle.dump(cur_model, open(cur_model_filename, "wb"))
            cur_model = pickle.load(open(cur_model_filename, "rb"))
            
            y_pred = cur_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1 score", f1)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for basic {cur_model_Name}")
            plt.savefig(f"confusion_matrix_basic_{cur_model_Name}.png")
            
            # Log artifacts (confusion matrix)
            mlflow.log_artifact(f"confusion_matrix_basic_{cur_model_Name}.png")

        
            train_df = mlflow.data.from_pandas(train_processed_data)
            test_df = mlflow.data.from_pandas(test_processed_data)
            
            mlflow.log_input(train_df, "train")  # Log training data
            mlflow.log_input(test_df, "test")  # Log test data

            # Log the current script file as an artifact in MLflow
            sign = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(cur_model, f"Best Model {cur_model_Name}", signature=sign)
            mlflow.set_tag("author", "anastasiarblv")
            mlflow.set_tag("model", cur_model_Name)
            mlflow.set_tag("model tuning", "No: basic model")
            mlflow.log_artifact(__file__)           # Log the source code file for reference
