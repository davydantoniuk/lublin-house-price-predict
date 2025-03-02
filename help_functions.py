import pandas as pd  
import numpy as np  
import re 
import sys
import warnings  
import joblib  
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import boxcox
from scipy.special import inv_boxcox

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  
import seaborn as sns  

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

from sklearn.impute import KNNImputer
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.optim as optim

import shap  
import lime  
import lime.lime_tabular

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")  

# Function to evaluate a model using multiple metrics
def evaluate_model(y_true, y_pred):
    """
    Evaluates a regression model using multiple metrics.
    
    Parameters:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
    
    Outputs:
        Prints R² Score, MAE, MSE, RMSE, MAPE and RMSLE.
    """
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    
    return r2, mae, mse, rmse, mape, rmsle


# Function to check normality using histogram, Q-Q plot, and statistical tests
def check_normality(data, columns, alpha=0.05):
    """
    This function checks the normality of specified columns in a DataFrame using histograms, Q-Q plots, 
    and statistical tests (Shapiro-Wilk and D'Agostino's K² tests). It also determines whether to reject 
    the null hypothesis that the data is normally distributed based on the p-values from these tests.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data to be tested.
    columns (list): A list of column names to check for normality.
    alpha (float): The significance level to use for the hypothesis tests (default is 0.05).

    Returns:
    None
    """
    for col in columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        sns.histplot(data[col], bins=30, kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram of {col}")
        
        # Q-Q Plot
        stats.probplot(data[col], dist="norm", plot=axes[1])
        axes[1].set_title(f"Q-Q Plot of {col}")
        
        plt.show()
        
        # Statistical Tests
        shapiro_test = stats.shapiro(data[col].dropna())  # Shapiro-Wilk Test
        dagostino_test = stats.normaltest(data[col].dropna())  # D'Agostino's K² Test

        print(f"Normality Tests for {col}:")
        print(f"  Shapiro-Wilk Test: W={shapiro_test[0]:.4f}, p-value={shapiro_test[1]:.4f}")
        print(f"  D'Agostino's K² Test: Stat={dagostino_test[0]:.4f}, p-value={dagostino_test[1]:.4f}")
        
        # Decision to reject null hypothesis
        if shapiro_test[1] < alpha:
            print(f"  Reject null hypothesis for Shapiro-Wilk Test at alpha={alpha}")
        else:
            print(f"  Fail to reject null hypothesis for Shapiro-Wilk Test at alpha={alpha}")
        
        if dagostino_test[1] < alpha:
            print(f"  Reject null hypothesis for D'Agostino's K² Test at alpha={alpha}")
        else:
            print(f"  Fail to reject null hypothesis for D'Agostino's K² Test at alpha={alpha}")
        
        print("-" * 50)
        
def check_feature_importance(model, X_test, num_observations=1):
    """
    This function checks the feature importance of a trained model using SHAP values.
    
    Parameters:
    model (sklearn model): The trained model to explain.
    X_test (DataFrame): The test set features.
    num_observations (int): The number of random observations to predict and explain. Default is 1.
    
    Returns:
    None: Displays SHAP waterfall plots for the specified number of random observations.
    
    Usage:
    check_feature_importance(rf_model2, X_test, num_observations=3)
    """
    for _ in range(num_observations):
        # Select a random observation from the test set
        random_index = np.random.randint(0, X_test.shape[0])
        random_observation = X_test.iloc[random_index:random_index+1]

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for the random observation
        shap_values = explainer.shap_values(random_observation)
        
        # Plot SHAP waterfall plot for the random observation
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                             base_values=explainer.expected_value[0],
                                             data=random_observation.values[0],
                                             feature_names=random_observation.columns))
        
# Function to detect outliers using the IQR method
def detect_outliers_iqr(df, column):
    '''
    Detects outliers in a column using the IQR method.

    - Outliers: Values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR.
    - Returns a DataFrame with outliers.

    Example:
    outliers = detect_outliers_iqr(df, 'Price')
    '''
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]


# Function to Winsorize a numeric column (cap extreme values)
def winsorize_series(series, lower_quantile=0.01, upper_quantile=0.99):
    '''
    Caps extreme values at given percentiles (default 1%-99%).

    - Values below/above limits are replaced with 1st/99th percentile.
    - Prevents extreme values from skewing results.

    Example:
    df['Year'] = winsorize_series(df['Year'])
    '''
    lower_limit = series.quantile(lower_quantile)
    upper_limit = series.quantile(upper_quantile)
    return series.clip(lower=lower_limit, upper=upper_limit)

# Function to compute k-distance plot
def get_kdist_plot(X, k):
    '''
    This function computes the k-distance plot for a dataset.
    https://stackoverflow.com/questions/15050389/estimating-choosing-optimal-hyperparameters-for-dbscan/15063143#15063143
    '''
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    
    # Compute distances to k-nearest neighbors
    distances, indices = nbrs.kneighbors(X)
    
    # Sort distances for better visualization
    distances = np.sort(distances[:, k - 1], axis=0)
    
    # Plot k-distance graph
    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.xlabel('Points in the dataset', fontsize=12)
    plt.ylabel(f'Sorted {k}-nearest neighbor distance', fontsize=12)
    plt.title(f'K-distance plot for DBSCAN (k={k})')
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    
# Function to compute sample weights for categorical features
def compute_weights(data, column):
    counts = data[column].value_counts()
    total_count = len(data)
    return total_count / counts

# Function to compare model performance on a given metric
def plot_metric_comparison(metric_values, metric_name):
    """
    Plots a bar chart comparing model performance on a given metric.
    
    Parameters:
        metric_values (dict): Dictionary where keys are model names and values are metric scores.
        metric_name (str): Name of the metric to be displayed as the plot title.
    """

    # Sort models by metric value (descending order)
    sorted_metrics = dict(sorted(metric_values.items(), key=lambda item: item[1], reverse=True))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot bar chart
    ax.barh(list(sorted_metrics.keys()), list(sorted_metrics.values()), color='royalblue')
    
    # Add labels and title
    ax.set_xlabel(metric_name)
    ax.set_title(f"{metric_name} Comparison")
    
    # Show grid for readability
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Display the plot
    plt.show()