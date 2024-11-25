import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub


dagshub.init(repo_owner='anastasiarblv', repo_name='mlops_project_github', mlflow=True)  # из dagshub_test.py
mlflow.set_experiment("Experiment 1")  # Name of the experiment in MLflow
mlflow.set_tracking_uri("https://dagshub.com/anastasiarblv/mlops_project_github.mlflow") # из dagshub_test.py


data = pd.read_csv(r"C:\Users\honor\Desktop\water_potability.csv") # адрес файл с рабочего стоала компа


from sklearn.model_selection import train_test_split
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
y_train = train_processed_data["Potability"]  


from sklearn.ensemble import RandomForestClassifier
import pickle
n_estimators = 100  
# Start a new MLflow run for tracking the experiment
with mlflow.start_run():
    # Initialize and train the Random Forest model
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    # Save the trained model to a file using pickle
    pickle.dump(clf, open("model.pkl", "wb"))
    X_test = test_processed_data.iloc[:, 0:-1].values  
    y_test = test_processed_data.iloc[:, -1].values  

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # Load the saved model for prediction
    model = pickle.load(open('model.pkl', "rb"))
    # Predict the target for the test data
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)  
    precision = precision_score(y_test, y_pred)  
    recall = recall_score(y_test, y_pred)  
    f1 = f1_score(y_test, y_pred)  

    # Log metrics to MLflow for tracking
    mlflow.log_metric("acc", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score", f1)

    # Log the number of estimators used as a parameter
    mlflow.log_param("n_estimators", n_estimators)

    # Generate a confusion matrix to visualize model performance
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True)  # Visualize confusion matrix
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save the confusion matrix plot as a PNG file
    plt.savefig("confusion_matrix.png")

    # Log the confusion matrix image to MLflow
    mlflow.log_artifact("confusion_matrix.png")

    # Log the trained model to MLflow
    mlflow.sklearn.log_model(clf, "RandomForestClassifier")

    # Log the source code file for reference
    mlflow.log_artifact(__file__)

    # Set tags in MLflow to store additional metadata
    mlflow.set_tag("author", "datathinkers")
    mlflow.set_tag("model", "GB")

    # Print out the performance metrics for reference
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)