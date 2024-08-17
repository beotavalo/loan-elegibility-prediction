# Step 0:  your env must have, MLFlow and Prefect installed
# Step 1:  Define your goal: The main goal is to convert your Jupyter/Colab notebook into an MLOps workflow.  The output a file named:  orchestrate.py
# orchestrate.py will be called by Prefect and it will be the artifact that you use to automate your ML training, deploy, etc.
# Step 2: Convert Notebook into a script
# Start with the needed imports
# 2a

import pandas as pd
import numpy as np
from comet_ml import Experiment
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

import os
from dotenv import load_dotenv
# import mlflow
# import xgboost as xgb
from prefect import flow, task

# 2b: Identify each task that is in the notebook
# this task comes from the notebook and it reads the data


@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_csv(filename)
    # Drop the Loan_ID feature
    df = df.drop(['Loan_ID'], axis=1)

    return df


@task
def feature_engineering(df_train: pd.DataFrame):
    # Filling missing values
    df_train['Gender'].fillna(df_train['Gender'].mode()[0], inplace=True)
    df_train['Married'].fillna(df_train['Married'].mode()[0], inplace=True)
    df_train['Dependents'].fillna(
        df_train['Dependents'].mode()[0], inplace=True)
    df_train['Self_Employed'].fillna(
        df_train['Self_Employed'].mode()[0], inplace=True)
    df_train['Credit_History'].fillna(
        df_train['Credit_History'].mode()[0], inplace=True)
    df_train['Loan_Amount_Term'].fillna(
        df_train['Loan_Amount_Term'].mode()[0], inplace=True)
    df_train['LoanAmount'].fillna(
        df_train['LoanAmount'].median(), inplace=True)

    # Categorical variables encoding
    df_train.Loan_Status = df_train.Loan_Status.replace({'Y': 1, 'N': 0})
    df_train.Gender = df_train.Gender.replace({'Male': 1, 'Female': 0})
    df_train.Married = df_train.Married.replace({'Yes': 1, 'No': 0})
    df_train.Self_Employed = df_train.Self_Employed.replace(
        {'Yes': 1, 'No': 0})
    df_train.Education = df_train.Education.replace(
        {'Graduate': 1, 'Not Graduate': 0})
    df_train.Property_Area = df_train.Property_Area.replace(
        {'Rural': 1, 'Semiurban': 2, 'Urban': 3})
    df_train.Dependents = df_train.Dependents.replace(
        {'0': 0, '1': 1, '2': 2, '3+': 3})
    # Add new feature
    df_train['Total_Income'] = df_train['ApplicantIncome'] + \
        df_train['CoapplicantIncome']
    df_train['EMI'] = df_train['LoanAmount']/df_train['Loan_Amount_Term']
    df_train['Balance Income'] = df_train['Total_Income'] - \
        (df_train['EMI']*1000)

    # Outliers treatment
    df_train['LoanAmount_log'] = np.log(df_train['LoanAmount'])
    df_train['Total_Income_log'] = np.log(df_train['Total_Income'])

    return df_train


@task
def feature_selection(df_train: pd.DataFrame):
    """
    First, I specify the model
    Then I use the selectFromModel object from sklearn, which
    will select the features which coefficients are non-zero"""
    # Capture the dependent feature
    y = df_train[['Loan_Status']]

    # Select independent feature from dataset
    X = df_train.drop(['Loan_Status'], axis=1)
    feature_sel_model = SelectFromModel(estimator=LogisticRegression())
    feature_sel_model.fit(X, y)
    selected_feat = X.columns[(feature_sel_model.get_support())]
    X = X[selected_feat]
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    return x_train, x_val, y_train, y_val


@task(log_prints=True)
def train_best_model(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame
) -> None:
    """train a model with best hyperparams and write everything out"""
    load_dotenv()
    # Access the API key
    COMET_API_KEY = os.environ.get('COMET_API_KEY')
    exp = Experiment(
        api_key=COMET_API_KEY,
        project_name="loan-eligibility",
        workspace="beotavalo"
    )
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    parameters = model.get_params()
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    MSE = mean_squared_error(y_val, y_pred)
    matrix = confusion_matrix(y_val, y_pred)
    print(f'The Accuracy of the model is: {accuracy}')
    print(f'The MSE of the model is : {MSE}')
    labels = ["N", "Y"]
    # Log the parameters and metrics
    exp.log_parameters(parameters)
    exp.log_metrics({"accuracy": accuracy, "MSE": MSE, 'recall': recall})

    # Log the confusion matrix to Comet
    exp.log_confusion_matrix(matrix=matrix, labels=labels)

    # Log the model
    from joblib import dump
    dump(model, "/workspaces/loan-elegibility-prediction/src/models/le_model.pickle")

    # Log the model
    exp.log_model("le_model.pickle",
                  "/workspaces/loan-elegibility-prediction/src/models")

    # Register the model
    exp.register_model(model_name="le_model.pickle", version="1.0.4")
    exp.end()
    return None


@flow
def main_flow(
    train_path: str = "/workspaces/loan-elegibility-prediction/data/raw/loan-train.csv"
) -> None:
    """The main training pipeline"""

    # MLflow settings
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # mlflow.set_experiment("nyc-taxi-experiment")

    # mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
    # set the experiment id
    # mlflow.set_experiment(experiment_id="0")
    # mlflow.autolog()

    # Load
    df_train = read_data(train_path)
    # df_val = read_data(val_path)
    df_train = feature_engineering(df_train)

    # Transform
    X_train, X_val, y_train, y_val = feature_selection(df_train)

    # Train
    train_best_model(X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    main_flow()
