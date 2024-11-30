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
mlflow.set_experiment("1_experiment__9999: tuning models")  # Name of the experiment in MLflow
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
RandomForestClassifier_space = {'n_estimators': [100, 200, 300, 500, 1000], 'max_depth': [None, 4, 5, 6, 10]}
LogisticRegression_space = {'penalty':['l1', 'l2'], 'C': np.logspace(0, 4, 10), 'max_iter': [50,75,100,200,300,400,500,700,800,1000]}

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
with mlflow.start_run(run_name="tuning: Water_Potability_Models_Experiment"):
    # Iterate over each model in the dictionary
    for cur_model_Name, cur_model, param in nabor:
        # Start a child run within the parent run for each individual model
        #!!!random_search = RandomizedSearchCV(estimator=cur_model, param_distributions=param, verbose=0, cv=10, n_iter=10, scoring ='accuracy', random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(estimator=cur_model, param_distributions=param, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)
        with mlflow.start_run(run_name=cur_model_Name, nested=True): 
            search_model = random_search.fit(X_train, y_train)
            for i in range(len(random_search.cv_results_['params'])):
                with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
                    mlflow.log_params(random_search.cv_results_['params'][i])  # Log the parameters
                    mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])  # Log the mean test score
            best_cur_model = search_model.best_estimator_
            best_params_cur_model = search_model.best_params_
            print("Best parameters found: ", best_params_cur_model)
            mlflow.log_params(best_params_cur_model)

            # Train the model using the best parameters identified by RandomizedSearchCV
            best_cur_model.fit(X_train, y_train)
            # Save the trained model to a file for later use
            best_cur_model_filename = f"tuning_{cur_model_Name}.pkl"
            pickle.dump(best_cur_model, open(best_cur_model_filename, "wb"))
            # Load the saved model from the file
            best_cur_model = pickle.load(open(best_cur_model_filename, "rb"))
            
            # Make predictions on the test set using the loaded model
            y_pred = best_cur_model.predict(X_test)
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
            plt.title(f"Confusion Matrix for {cur_model_Name}")
            plt.savefig(f"confusion_matrix_tuning_{cur_model_Name}.png")
            
            # Log artifacts (confusion matrix)
            mlflow.log_artifact(f"confusion_matrix_tuning_{cur_model_Name}.png")

        
            train_df = mlflow.data.from_pandas(train_processed_data)
            test_df = mlflow.data.from_pandas(test_processed_data)
            
            mlflow.log_input(train_df, "train")  # Log training data
            mlflow.log_input(test_df, "test")  # Log test data

            # Log the current script file as an artifact in MLflow
            sign = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(best_cur_model, f"Best Model {cur_model_Name}", signature=sign)
            mlflow.set_tag("author", "anastasiarblv")
            mlflow.set_tag("model", cur_model_Name)
            mlflow.set_tag("model tuning", "Yes: RandomizedSearchCV")
            mlflow.log_artifact(__file__)           # Log the source code file for reference



