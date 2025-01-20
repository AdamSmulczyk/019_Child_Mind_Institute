#!/usr/bin/env python
# coding: utf-8

# In[7]:

# import sys
# sys.path.append(r'd:\pycharmprojects\moj_pythonproject\kaggle\competitions\019a_CMI')
# from data_processor_cmi import make_pipeline
from data_processor_cmi import *
from dotenv import load_dotenv

import os
from datetime import datetime
import base64
import requests
import pandas as pd
import numpy as np
from scipy.stats import boxcox, stats
from sklearn.preprocessing import  MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder 
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold, KFold, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, matthews_corrcoef, average_precision_score,f1_score,cohen_kappa_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.tree import  DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from collections import Counter
from yellowbrick.classifier import ROCAUC
import optuna
from abc import ABC, abstractmethod
from typing import Tuple, Type
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
# DATA VISUALIZATION
# ------------------------------------------------------
# import skimpy
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIGURATIONS
# ------------------------------------------------------
warnings.filterwarnings('ignore')
pd.set_option('float_format', '{:.3f}'.format)


class Model(ABC):
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for compatibility with multi-class scoring."""
        return self.model.predict_proba(X)
    
    def fit(self, X, y):
        """Fit method for compatibility with sklearn's cross_val_score."""
        self.train(X, y)
        return self
    
    def score(self, X, y, scoring: str = 'r2'):

        if scoring == 'r2':
            # Use standard accuracy for classification.
            predictions = self.predict(X)
            return r2_score(y, predictions)
        elif scoring == 'roc_auc_ovo':
            # Use ROC AUC score for multi-class problems.
            probabilities = self.predict_proba(X)
            return roc_auc_score(y, probabilities, multi_class='ovo', average='macro')
        else:
            raise ValueError(f"Unsupported scoring method: {scoring}. Supported metrics are 'accuracy' and 'roc_auc_ovo'.")
            
class LGBModel(Model, BaseEstimator, RegressorMixin):
    """LGBModel Regressor model with extended parameter support."""
    def __init__(self, **kwargs):
        self.model = LGBMRegressor(**kwargs)            
    
class XGBoostModel(Model, BaseEstimator, RegressorMixin):
    """XGBoost Regressor model with extended parameter support."""
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)
        
class CatBoostModel(Model, BaseEstimator, RegressorMixin):
    """CatBoost Regressor model with extended parameter support."""
    def __init__(self, **kwargs):
        self.model = CatBoostRegressor(**kwargs)
        
class RandomForestModel(Model, BaseEstimator, RegressorMixin):
    """Random Forest Regressor model."""
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)   

class VotingModel(Model, BaseEstimator, RegressorMixin):
    """Voting Regressor combining RFC_1 and XGB_2."""
    def __init__(self, estimators, voting='soft', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.model = VotingRegressor(estimators=self.estimators, voting=self.voting, weights=self.weights)     
        
class ModelFactory:
    """Factory to create model instances."""
    @staticmethod
    def get_model(model_name: str, **kwargs) -> Model:
        model_class = globals()[model_name]
        return model_class(**kwargs)

class Workflow_8:
    """Main workflow class for model training and evaluation."""
    def run_workflow(self, 
                     model_name: str, 
                     model_kwargs: dict, 
                     X: pd.DataFrame, 
                     y: pd.Series,
                     test_size: float, 
                     random_state: int,
                     scoring: str = 'r2'
                    ) -> None:
        """
        Main entry point to run the workflow:
        - Splits the data.
        - Trains the model.
        - Evaluates the model.
        """
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)

        if model_name == 'VotingModel':
            model = VotingModel(**model_kwargs)
        else:
            model = ModelFactory.get_model(model_name, **model_kwargs)

        pipe = make_pipeline(preprocessor_1, model)
        pipe.fit(X_train, y_train)


        
        load_dotenv()
#         github_token = os.getenv("GITHUB_TOKEN")
#         print(f"Loaded token: {github_token}")
        github_token = os.getenv("CMI_GITHUB_TOKEN")
        if not github_token:
            raise ValueError("CMI_GITHUB_TOKEN is not set. Please configure it in your environment.")
            
            
            # Test the GitHub token
        self.test_github_token(github_token)
        
        
        github_repo_url = "https://github.com/AdamSmulczyk/019_Child_Mind_Institute/performance_reports"
        
        
        results = self.evaluate_model(pipe, X_train, X_test, y_train, y_test, scoring)       
        print("Model Evaluation Results:")
        print(results.to_string())  
        self.save_results_to_file(results,
                          save_dir=r"C:\Users\adams\OneDrive\Dokumenty\Python",
                          prefix="cmi_evaluate",
                          file_type="csv",
                          github_token=github_token, 
                          github_repo_url=github_repo_url
                         )

    @staticmethod
    def test_github_token(github_token: str) -> None:
        """
        Tests the GitHub token by making an authenticated API request.

        Parameters:
        - github_token (str): The GitHub personal access token to test.

        Raises:
        - ValueError: If the token is invalid or the API request fails.
        """
        api_url = "https://api.github.com/user"
        headers = {"Authorization": f"Bearer {github_token}"}

        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            user_data = response.json()
            print(f"Token is valid. Authenticated as: {user_data['login']}")
        else:
            raise ValueError(f"Invalid GitHub token or API error: {response.status_code} - {response.text}")

    def split_data(self, 
                   X: pd.DataFrame, 
                   y: pd.Series, 
                   test_size: float, 
                   random_state: int
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def evaluate_model(self, 
                       model: Model, 
                       X_train: pd.DataFrame, 
                       X_test: pd.DataFrame, 
                       y_train: pd.Series, 
                       y_test: pd.Series,
                       scoring: str) -> pd.DataFrame:

        """
        Evaluate the model using custom metrics.
        
        Parameters:
        - model: Trained model to evaluate.
        - X_train, X_test: Feature datasets for training and testing.
        - y_train, y_test: Target datasets for training and testing.
        
        Returns:
        - pd.DataFrame: DataFrame containing evaluation metrics for train and test sets.
        """
        model_train_score = model.score(X_train, y_train)
        model_test_score = model.score(X_test, y_test)
        

        y_pred_1_train = model.predict(X_train)
        y_pred_1_valid = model.predict(X_test)
        print(f"\
Train Score = {model_train_score *100}%\n\
Valid Score = {model_test_score*100}%\n\
Mean Squared Error = {mean_squared_error(y_test, y_pred_1_valid)}\n\
Mean Absolute Error = {mean_absolute_error(y_test, y_pred_1_valid)}")
        
        """Evaluate the model using K-Fold cross-validation."""
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        n_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring)  
        print(f'R2 Score = {np.mean(n_scores)}')
            
        def create_measures(y,y_pred)-> pd.Series:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            n_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

            d = {'r2 score': [round(r2_score(y, y_pred),4)],
                 'KFold r2 score': [round(np.mean(n_scores),4)],
                 'Mean Squared Error': [mean_squared_error(y, y_pred)],
                 'Mean Absolute Error': [mean_absolute_error(y, y_pred)],
                }

            return pd.DataFrame.from_dict(d)

        def calculating_metrics(X_train, X_valid, y_train, y_valid, model):
            train = create_measures(y_train, model.predict(X_train))
            valid = create_measures(y_valid, model.predict(X_valid))

            return pd.concat([train,valid]).set_index([pd.Index(['TRAIN', 'VALID'])]) 


        # Calculate metrics for training and validation data
        results = calculating_metrics(X_train, X_test, y_train, y_test, model)
        return results
    
    @staticmethod
    def save_results_to_file(data, 
                             save_dir: str, 
                             prefix: str, 
                             file_type: str = "csv",
                             github_token: str = None, 
                             github_repo_url: str = None) -> None:
        """
        Save data (results or plots) to a file.

        Parameters:
        - data: The data to save (pd.DataFrame for CSV or matplotlib figure for plots).
        - save_dir: Directory to save the file.
        - prefix: File prefix for naming the output file.
        - file_type: Type of file to save ("csv" for results, "png" for plots).
        """
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if file_type == "csv":
            file_name = f"{prefix}_{timestamp}.csv"
            output_path = os.path.join(save_dir, file_name)
            print(f"Saving results to: {output_path}")
            data.to_csv(output_path, index=True)
        elif file_type == "png":
            file_name = f"{prefix}_{timestamp}.png"
            output_path = os.path.join(save_dir, file_name)
            print(f"Saving plot to: {output_path}")
            data.savefig(output_path, bbox_inches='tight')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
         
        if github_token and github_repo_url:
            Workflow_8.upload_to_github(output_path, github_repo_url, github_token)
            


    @staticmethod
    def upload_to_github(output_path: str, 
                         github_repo_url: str, 
                         github_token: str, 
                         branch: str = "main",
                         commit_message: str = "Add file") -> None:
        """
        Uploads a file to a GitHub repository.

        Parameters:
        - output_path (str): Path to the file on the local disk.
        - github_repo_url (str): URL of the GitHub repository (e.g., 'https://github.com/User/RepoName').
        - github_token (str): GitHub personal access token for authentication.
        - branch (str): Branch to which the file should be uploaded. Default is 'main'.
        - commit_message (str): Commit message for the upload. Default is 'Add file'.
        """
        # Parse the repository URL
        url_parts = github_repo_url.rstrip("/").split("/")
        if len(url_parts) < 5:
            raise ValueError(f"Invalid GitHub repository URL: {github_repo_url}")

        user = url_parts[3]  # GitHub username
        repo_name = url_parts[4]  # Repository name

        # Define folder_path as empty unless specified
        folder_path = ""  
        if len(url_parts) > 5:
            folder_path = "/".join(url_parts[5:])

        # Determine the API URL
        file_name = os.path.basename(output_path)
        api_url = f"https://api.github.com/repos/{user}/{repo_name}/contents/{folder_path}/{file_name}" if folder_path else f"https://api.github.com/repos/{user}/{repo_name}/contents/{file_name}"

        # Read the file content
        with open(output_path, "rb") as file:
            content = base64.b64encode(file.read()).decode("utf-8")

        # Prepare the payload for the request
        payload = {
            "message": commit_message,
            "branch": branch,
            "content": content,
        }

        # Prepare the headers
        headers = {"Authorization": f"token {github_token}"}

        # Send the PUT request to GitHub API
        response = requests.put(api_url, json=payload, headers=headers)

        # Handle response
        if response.status_code == 201:
            print(f"File '{output_path}' successfully uploaded to GitHub at '{github_repo_url}'.")
        elif response.status_code == 404:
            print(f"Failed to upload file: {response.status_code} - Repository or path not found.")
        elif response.status_code == 422:
            print(f"File '{output_path}' already exists in the repository.")
        else:
            print(f"Failed to upload file: {response.status_code} - {response.text}")
            
            
#     @staticmethod
#     def upload_to_github(output_path: str, 
#                          github_repo_url: str, 
#                          github_token: str, 
#                          branch: str = "main",
#                          commit_message: str = "Add file") -> None:
#         """
#         Uploads a file to a GitHub repository.

#         Parameters:
#         - output_path (str): Path to the file on the local disk.
#         - github_repo_url (str): URL of the GitHub repository (e.g., 'https://github.com/User/RepoName/tree/main/folder').
#         - github_token (str): GitHub personal access token for authentication.
#         - branch (str): Branch to which the file should be uploaded. Default is 'main'.
#         - commit_message (str): Commit message for the upload. Default is 'Add file'.
#         """
#         # Construct the API URL for the target file
#         url_parts = github_repo_url.rstrip("/").split("/")
#         repo_name = url_parts[4]  # Repository name
#         user = url_parts[3]       # GitHub username
#         folder_path = "/".join(url_parts[6:])  # Folder path inside the repository       
#         file_name = os.path.basename(output_path)
#         api_url = f"https://api.github.com/repos/{user}/{repo_name}/contents/{folder_path}/{file_name}"

#         # Read the file content
#         with open(output_path, "rb") as file:
#             content = base64.b64encode(file.read()).decode("utf-8")

#         # Prepare the payload for the request
#         payload = {
#             "message": commit_message,
#             "branch": branch,
#             "content": content,
#         }

#         # Prepare the headers
#         headers = {"Authorization": f"Bearer {github_token}"}

#         # Send the PUT request to GitHub API
#         response = requests.put(api_url, json=payload, headers=headers)

#         # Handle response
#         if response.status_code == 201:
#             print(f"File '{output_path}' successfully uploaded to GitHub at '{github_repo_url}'.")
#         elif response.status_code == 404:
#             print(f"Failed to upload file: {response.status_code} - Repository or path not found.")
#         elif response.status_code == 422:
#             print(f"File '{output_path}' already exists in the repository.")
#         else:
#             print(f"Failed to upload file: {response.status_code} - {response.text}")