import pandas as pd  
import numpy as np  
import re 
import sys
import warnings  
import os
import joblib  
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import statsmodels.api as sm

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  
import seaborn as sns  
from IPython.display import display

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor

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

from smogn import smoter
import smogn
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
        
def check_feature_importance(model, X_test, num_observations=1, random_seed=None):
    """
    This function checks the feature importance of a trained model using SHAP values.
    
    Parameters:
    model (sklearn model): The trained model to explain.
    X_test (DataFrame): The test set features.
    num_observations (int): The number of random observations to predict and explain. Default is 1.
    random_seed (int, optional): Random seed for reproducibility. Default is None.
    
    Returns:
    None: Displays SHAP waterfall plots for the specified number of random observations.
    
    Usage:
    check_feature_importance(rf_model2, X_test, num_observations=3, random_seed=42)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
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


# Function to Winsorize a numeric column and return limits
def winsorize_series(series, lower_quantile=0.01, upper_quantile=0.99):
    '''
    Caps extreme values at given percentiles (default 1%-99%), using nearest integer values.

    - Calculates quantiles using nearest interpolation to ensure integer limits.
    - Clips values to these integer limits to maintain the original integer dtype.
    - Returns the transformed series and the computed limits for consistency across datasets.
    '''
    lower_limit = series.quantile(lower_quantile, interpolation='nearest')
    upper_limit = series.quantile(upper_quantile, interpolation='nearest')
    
    return series.clip(lower=lower_limit, upper=upper_limit).astype(series.dtype), lower_limit, upper_limit

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
    
# Define parameter grid for Random Forest
PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}


def preprocess_data(method, X_train, X_val, X_test):
    '''
    Prepares the dataset based on the chosen encoding method.

    Parameters:
    - method (str): Encoding method to apply.
        - "LE"  → Label Encoding for categorical features.
        - "OHE" → One-Hot Encoding for categorical features.
        - "OHE-W" → One-Hot Encoding + Sample Weighting.
    - X_train, X_val, X_test (DataFrame): Training, validation, and test sets.

    Returns:
    - X_train (DataFrame): Processed training features.
    - X_test (DataFrame): Processed test features.
    - sample_weight (Series or None): Sample weights (for OHE-W) or None.
    '''
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

    # Store Floor & Rooms before encoding for weighting
    if method == "OHE-W":
        floor_col = X_train['Floor'].copy()
        rooms_col = X_train['Rooms'].copy()

    if method == "LE":  # Label Encoding
        for col in ['Floor', 'Region']:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_val[col] = le.transform(X_val[col])
            X_test[col] = le.transform(X_test[col])

    elif method in ["OHE", "OHE-W"]:  # One-Hot Encoding
        X_train, X_val, X_test = [pd.get_dummies(df, columns=['Floor', 'Region'], drop_first=True) for df in [X_train, X_val, X_test]]
        for df in [X_val, X_test]:  
            for col in set(X_train.columns) - set(df.columns):
                df[col] = 0
            df = df[X_train.columns]

    if method == "OHE-W":  # Compute sample weights for weighted model
        room_w = rooms_col.value_counts().to_dict()
        floor_w = floor_col.value_counts().to_dict()
        sample_weight = rooms_col.map(room_w) * floor_col.map(floor_w)
        return X_train, X_test, sample_weight

    return X_train, X_test, None  # Return data with no sample weight


def train_evaluate(X_train, X_test, y_train, y_test, sample_weight, param_grid=PARAM_GRID):
    '''
    Trains a Random Forest model using Grid Search CV and evaluates performance.

    Parameters:
    - X_train (DataFrame): Training feature set.
    - X_test (DataFrame): Test feature set.
    - y_train (Series): Target values for training.
    - y_test (Series): Target values for testing.
    - sample_weight (Series or None): Sample weights for training.
    - param_grid (dict): Hyperparameter grid for GridSearchCV.

    Returns:
    - evaluation_results (dict): Model performance metrics.
    - best_params (dict): Best hyperparameters from Grid Search.
    '''
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train, sample_weight=sample_weight)
    best_model = grid_search.best_estimator_
    return evaluate_model(y_test, best_model.predict(X_test)), grid_search.best_params_

# Function to prepare data for stacking model
def prepare_data_for_stacking(X_tree, y_tree, use_combined_train_val=False, models=['rf', 'cat', 'xgb']):
    """
    Prepares data for stacking model by applying transformations, encoding categorical features,
    and ensuring consistency across train and test sets.

    Parameters:
    - X_tree (pd.DataFrame): Feature dataset
    - y_tree (pd.Series): Target variable
    - use_combined_train_val (bool): If True, merges training and validation sets before training
    - models (list): List of models to prepare data for (options: 'rf', 'cat', 'xgb')

    Returns:
    - Dictionary containing processed train and test datasets
    - sample_weight_rf (Sample weights for RF model)
    """
    
    # === 1. Train-Validation-Test Split ===
    X_train, X_temp, y_train, y_temp = train_test_split(X_tree, y_tree, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    if use_combined_train_val:
        # Combine Training and Validation Sets
        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)

    # === 2. Apply Transformations ===
    # Box-Cox transformation
    X_train['Area'], area_lambda = stats.boxcox(X_train['Area'])
    X_test['Area'] = stats.boxcox(X_test['Area'], lmbda=area_lambda)

    # Winsorization
    X_train['Year'], year_lower, year_upper = winsorize_series(X_train['Year'])
    X_test['Year'] = X_test['Year'].clip(lower=year_lower, upper=year_upper).astype(X_test['Year'].dtype)
    
    # KNN Imputation - Model performed better without imputation
    # floor_mapping = {
    # 'parter': 0,
    # '1 piętro': 1,
    # '2 piętro': 2,
    # '3 piętro': 3,
    # '4 piętro': 4,
    # '5+ piętro': 5,  
    # np.nan: np.nan  # Keep NaNs for imputation
    # }
    
    # X_train['Floor'] = X_train['Floor'].map(floor_mapping)
    # X_test['Floor'] = X_test['Floor'].map(floor_mapping)

    # knn_imputer = KNNImputer(n_neighbors=5) # With 5 neighbors model performed the best
    
    # num_cols = ['Area', 'Elevator', 'Year', 'Rooms', 'Floor']
    # X_train[num_cols] = knn_imputer.fit_transform(X_train[num_cols])
    # X_test[num_cols] = knn_imputer.transform(X_test[num_cols])

    # X_train['Floor'] = np.round(X_train['Floor']).astype(int)
    # X_test['Floor'] = np.round(X_test['Floor']).astype(int)

    # reverse_floor_mapping = {v: k for k, v in floor_mapping.items()}
    
    # X_train['Floor'] = X_train['Floor'].map(reverse_floor_mapping)
    # X_test['Floor'] = X_test['Floor'].map(reverse_floor_mapping)

    # === 3. Prepare Data for Each Model ===
    data_dict = {}
    if 'rf' in models:
        # One-Hot Encoding for RF
        X_train_rf = pd.get_dummies(X_train.copy(), columns=['Floor', 'Region'], drop_first=True, dtype='int32')
        X_test_rf = pd.get_dummies(X_test.copy(), columns=['Floor', 'Region'], drop_first=True, dtype='int32')

        # Ensure test set has same features as training set
        missing_cols_rf = set(X_train_rf.columns) - set(X_test_rf.columns)
        for col in missing_cols_rf:
            X_test_rf[col] = 0
        X_test_rf = X_test_rf[X_train_rf.columns]

        # Compute Sample Weights for RF
        floor_columns = [col for col in X_train_rf.columns if col.startswith('Floor_')]
        region_columns = [col for col in X_train_rf.columns if col.startswith('Region_')]
        categorical_weight_columns = floor_columns + region_columns

        weights_dict = {}
        for col in categorical_weight_columns:
            value_counts = X_train_rf[col].value_counts()
            weights_dict[col] = {val: len(X_train_rf) / count for val, count in value_counts.items()}

        sample_weight_rf = pd.Series(1, index=X_train_rf.index)
        for col in categorical_weight_columns:
            sample_weight_rf *= X_train_rf[col].map(weights_dict[col])

        data_dict['X_train_rf'] = X_train_rf
        data_dict['X_test_rf'] = X_test_rf
        data_dict['sample_weight_rf'] = sample_weight_rf

    if 'cat' in models:
        # Label Encoding for CatBoost
        X_train_cat, X_test_cat = X_train.copy(), X_test.copy()
        label_encoders = {}
        for col in ['Floor', 'Region']:
            le = LabelEncoder()
            X_train_cat[col] = le.fit_transform(X_train_cat[col])
            X_test_cat[col] = le.transform(X_test_cat[col])
            label_encoders[col] = le

        data_dict['X_train_cat'] = X_train_cat
        data_dict['X_test_cat'] = X_test_cat

    if 'xgb' in models:
        # One-Hot Encoding for XGBoost
        X_train_xgb = pd.get_dummies(X_train.copy(), columns=['Floor', 'Region'], drop_first=True, dtype='int32')
        X_test_xgb = pd.get_dummies(X_test.copy(), columns=['Floor', 'Region'], drop_first=True, dtype='int32')

        # Ensure test set has same features as training set
        missing_cols_xgb = set(X_train_xgb.columns) - set(X_test_xgb.columns)
        for col in missing_cols_xgb:
            X_test_xgb[col] = 0
        X_test_xgb = X_test_xgb[X_train_xgb.columns]

        data_dict['X_train_xgb'] = X_train_xgb
        data_dict['X_test_xgb'] = X_test_xgb

    data_dict['y_train'] = y_train
    data_dict['y_test'] = y_test

    return data_dict

# Create DataLoader 
def create_dataloaders(X_train, y_train, X_val, y_val, batch_size, cnn_input=False):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    
    # Reshape for CNN if needed (batch_size, channels, sequence_length)
    if cnn_input:
        X_train_tensor = X_train_tensor.view(X_train_tensor.shape[0], 1, -1)
        X_val_tensor = X_val_tensor.view(X_val_tensor.shape[0], 1, -1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Return sequence length for CNN model, feature size for FNN model
    input_size = X_train_tensor.shape[-1] if cnn_input else X_train_tensor.shape[1]
    
    return train_loader, val_loader, input_size

# Create test DataLoader
def create_test_loader(X_test, y_test, batch_size, cnn_input=False):
    # Convert to tensors (handle both DataFrame and Series input)
    X_test_tensor = torch.tensor(X_test.values if hasattr(X_test, 'values') else np.array(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else np.array(y_test), dtype=torch.float32).view(-1, 1)
    
    # Reshape for CNN if needed
    if cnn_input:
        X_test_tensor = X_test_tensor.view(X_test_tensor.shape[0], 1, -1)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

# Define FNN Model 
class HousePriceFNN(nn.Module):
    def __init__(self, input_size):
        super(HousePriceFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# Define CNN Model 
class HousePriceCNN(nn.Module):
    def __init__(self, input_size):
        super(HousePriceCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

# Define RMSLE Loss 
class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=0)
        return torch.sqrt(torch.mean((torch.log1p(y_pred) - torch.log1p(y_true)) ** 2))

# Define MAPE Loss 
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-7
        return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# Train Model 
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device):
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = torch.clamp(model(X_batch), min=0)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)

        # === Validation Step ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = torch.clamp(model(X_batch), min=0)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} | Best Val Loss: {best_val_loss:.4f}")
            break

        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return history

# Plot Training History
def plot_training_history(history, title="Training History"):
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label="Train Loss", color='blue')
    plt.plot(history['val_loss'], label="Validation Loss", color='orange')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot and compare model performance
def plot_model_performance(r2_scores, mae_scores, rmse_scores, mape_scores, rmsle_scores):
    models = list(r2_scores.keys())
    cmap = plt.get_cmap('tab20')
    model_colors = {model: cmap(i) for i, model in enumerate(models)}

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1], wspace=1.2, hspace=0.4)

    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1:3])
    ax2 = plt.subplot(gs[0, 3:5])
    ax3 = plt.subplot(gs[1, 0:2])
    ax4 = plt.subplot(gs[1, 3:5])

    axes = [ax0, ax1, ax2, ax3, ax4]

    # Configure all axes
    for ax in axes:
        ax.tick_params(axis='y', labelsize=9)
        ax.xaxis.set_tick_params(labelsize=8)

    # Helper function to plot each metric
    def plot_metric(ax, metric_name, metric_values):
        sorted_metrics = dict(sorted(metric_values.items(), key=lambda item: item[1], reverse=True))
        colors = [model_colors[model] for model in sorted_metrics.keys()]
        
        bars = ax.barh(list(sorted_metrics.keys()), list(sorted_metrics.values()), color=colors)
        ax.set_title(f"{metric_name} Comparison", fontsize=11, pad=10)
        ax.set_xlabel(metric_name, fontsize=10, labelpad=8)
        ax.grid(axis='x', linestyle=':', alpha=0.7)

        # Dynamic x-lim padding
        xmax = max(metric_values.values())
        ax.set_xlim(left=0, right=xmax * 1.15)

    # Plot all metrics
    metrics_data = [
        (ax0, "R² Score", r2_scores),
        (ax1, "MAE", mae_scores),
        (ax2, "RMSE", rmse_scores),
        (ax3, "MAPE", mape_scores),
        (ax4, "RMSLE", rmsle_scores)
    ]

    for ax, name, values in metrics_data:
        plot_metric(ax, name, values)

    # Final layout adjustments
    plt.subplots_adjust(top=0.92, bottom=0.08)
    plt.suptitle("Model Performance Comparison", fontsize=14)
    plt.show()
    
# Function to plot SHAP explanations
def plot_shap_explanations(model, X_test, model_type='cat', num_cases=4, seed=42):
    shap.initjs()

    # Fill missing values with median to handle NaNs
    X_test = X_test.fillna(X_test.median())

    # Randomly select different rows each time (use seed for reproducibility)
    X_sample = X_test.sample(n=num_cases, random_state=seed).reset_index(drop=True)

    # List to store SHAP explanations and data
    explanations = []

    # Monkey-patch to capture explanation data
    original_waterfall_plot = shap.waterfall_plot

    def capturing_plot(explanation, *args, **kwargs):
        explanations.append({
            'values': explanation.values,
            'base_value': explanation.base_values,
            'data': explanation.data,
            'feature_names': explanation.feature_names
        })
        plt.close()

    shap.waterfall_plot = capturing_plot

    # Generate SHAP explanations
    if model_type == 'rf':
        explainer = shap.TreeExplainer(model.named_estimators_['rf'])
        shap_values = explainer.shap_values(X_sample)
    elif model_type == 'cat':
        explainer = shap.TreeExplainer(model.named_estimators_['cat'])
        shap_values = explainer.shap_values(X_sample)
    else:
        raise ValueError("Unsupported model type. Use 'rf' or 'cat'.")

    # Generate explanations and capture them
    for i in range(num_cases):
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[i],
                             base_values=explainer.expected_value,
                             data=X_sample.iloc[i],
                             feature_names=X_sample.columns),
            show=False
        )

    # Restore original SHAP function
    shap.waterfall_plot = original_waterfall_plot

    # Configure plot parameters BEFORE creating the figure
    plt.rcParams.update({
        'font.size': 14,           
        'axes.titlesize': 16,        
        'axes.labelsize': 12,         
        'xtick.labelsize': 10,        
        'ytick.labelsize': 10,       
        'figure.autolayout': False    
    })

    # Create a LARGER figure with more spacing to avoid text overlap
    fig, axes = plt.subplots(2, 2, figsize=(30, 30), dpi=150)
    axes = axes.flatten()

    # Plot with enhanced spacing
    for i, exp_data in enumerate(explanations[:4]):
        exp = shap.Explanation(
            values=exp_data['values'],
            base_values=exp_data['base_value'],
            data=exp_data['data'],
            feature_names=exp_data['feature_names']
        )

        ax = axes[i]
        plt.figure(fig.number)
        plt.sca(ax)

        # Create plot with adjusted parameters
        shap.plots.waterfall(
            exp,
            show=False,
            max_display=10  # Limit number of features displayed to avoid clutter
        )

        # Custom adjustments
        ax.set_title(f'Case {i + 1}', pad=20, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=10, length=5)

        # Add breathing room around the plot
        plt.draw()

    # Adjust spacing between plots
    plt.tight_layout(pad=8)  
    plt.subplots_adjust(
        wspace=2,  
        hspace=0.7   
    )

    plt.show()


# ================================================================================================
# ===========================Functions to predict new observations================================
# ================================================================================================

# Expected columns after transformation
EXPECTED_COLUMNS = [
    'Area', 'Elevator', 'Year', 'Rooms',
    'Floor_2 piętro', 'Floor_3 piętro', 'Floor_4 piętro', 'Floor_5+ piętro', 'Floor_parter',
    'Region_Czechów Północny', 'Region_Czuby Północne', 'Region_Dziesiąta', 'Region_Kośminek',
    'Region_Other', 'Region_Ponikwoda', 'Region_Rury', 'Region_Sławin', 'Region_Wieniawa',
    'Region_Wrotków', 'Region_Węglin Południowy', 'Region_Śródmieście'
]

# Floor mapping and merging
floor_mapping = {
    'parter': 'Floor_parter',
    '1 piętro': 'Floor_1 piętro',
    '2 piętro': 'Floor_2 piętro',
    '3 piętro': 'Floor_3 piętro',
    '4 piętro': 'Floor_4 piętro',
    '5+ piętro': 'Floor_5+ piętro'
}

# High floors to merge into '5+ piętro'
high_floors = ['5 piętro', '6 piętro', '7 piętro', '8 piętro', '9 piętro', '10 piętro', '10+ piętro']

# Region mapping (low-frequency → 'Other')
all_regions = [
    'Rury', 'Czechów Północny', 'Wrotków', 'Czechów Południowy', 'Kośminek',
    'Wieniawa', 'Ponikwoda', 'Śródmieście', 'Bronowice', 'Węglin Południowy',
    'Dziesiąta', 'Tatary', 'Felin', 'Kalinowszczyzna', 'Sławin', 'Czuby Północne',
    'Konstantynów', 'Szerokie', 'Czuby Południowe', 'Stare Miasto',
    'Za Cukrownią', 'Zemborzyce', 'Węglin Północny', 'Hajdów-Zadębie'
]

low_frequency_regions = [
    'Bronowice', 'Stare Miasto', 'Kalinowszczyzna', 'Konstantynów', 'Tatary', 
    'Czuby Południowe', 'Felin', 'Szerokie', 'Za Cukrownią', 'Zemborzyce',
    'Węglin Północny', 'Hajdów-Zadębie'
]

# Store Box-Cox lambda and winsorization limits
area_lambda = joblib.load('model_components/boxcox_lambda.pkl')
year_lower, year_upper = joblib.load('model_components/winsorization_limits.pkl')

# Load training and test sets
X_train, y_train, X_test, y_test = joblib.load('model_components/data_split.pkl')

# Load sample weights
try:
    sample_weight = joblib.load('model_components/sample_weights.pkl')
except FileNotFoundError:
    sample_weight = None

# PREPARE INPUT DATA
def prepare_input_data(df):
    '''
    Prepares the input data for model prediction by applying transformations, encoding, 
    and handling missing values.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing user-provided data.

    Returns:
    - pd.DataFrame: Processed DataFrame ready for prediction.
    '''
    # 1. Fix data types 
    df['Area'] = df['Area'].astype(float)
    df['Elevator'] = df['Elevator'].astype(int)
    df['Year'] = df['Year'].astype(int)
    df['Rooms'] = df['Rooms'].astype(int)

    # 2. Handle 'Floor' values 
    df['Floor'] = df['Floor'].replace({'poddasze': '3 piętro', 'suterena': 'parter'})
    df['Floor'] = df['Floor'].replace(high_floors, '5+ piętro')

    # Initialize one-hot encoded floor columns
    for col in floor_mapping.values():
        df[col] = 0

    if df['Floor'].iloc[0] in floor_mapping:
        df[floor_mapping[df['Floor'].iloc[0]]] = 1
    
    # 3. Handle 'Region' values 
    region = df['Region'].iloc[0]
    if region in low_frequency_regions:
        region = 'Other'

    # Initialize one-hot encoded region columns
    for region_col in all_regions:
        df[f'Region_{region_col}'] = 0

    if f'Region_{region}' in df.columns:
        df[f'Region_{region}'] = 1

    # 4. Apply Box-Cox transformation for 'Area' 
    df['Area'] = (df['Area'] ** area_lambda - 1) / area_lambda if area_lambda != 0 else np.log(df['Area'])

    # 5. Winsorize 'Year' 
    df['Year'] = df['Year'].clip(lower=year_lower, upper=year_upper)

    # 6. Drop original categorical columns 
    df.drop(columns=['Floor', 'Region'], inplace=True)

    # 7. Ensure all expected columns are present 
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # 8. Order columns to match model training order 
    df = df[EXPECTED_COLUMNS]
    
    return df


# SHAP WATERFALL PLOT
def plot_shap_waterfall(model, input_data, model_type):
    '''
    Generates a SHAP waterfall plot to explain individual model predictions.

    Parameters:
    - model (object): The trained stacking model.
    - input_data (pd.DataFrame): Input data for prediction.
    - model_type (str): Type of base model ('rf' or 'cat').

    Returns:
    - matplotlib.figure.Figure: SHAP waterfall plot.
    '''
    shap.initjs()

    if model_type == 'rf':
        explainer = shap.TreeExplainer(model.named_estimators_['rf'])
        shap_values = explainer.shap_values(input_data)[0]
        base_value = explainer.expected_value[0]
    elif model_type == 'cat':
        explainer = shap.TreeExplainer(model.named_estimators_['cat'])
        shap_values = explainer.shap_values(input_data)[0]
        base_value = explainer.expected_value
    else:
        raise ValueError("Unsupported model type. Use 'rf' or 'cat'.")

    # Create SHAP waterfall plot
    shap_exp = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=input_data.iloc[0].values,
        feature_names=input_data.columns
    )

    fig = plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_exp, show=False)
    plt.close(fig)  

    return fig


# PREDICT FUNCTION
def predict_house_price(model, input_df, model_type='rf', confidence_level=0.95):
    '''
    Predicts house price based on input features and returns the prediction with 
    a confidence interval and a SHAP waterfall plot.

    Parameters:
    - model (object): Trained stacking model (RandomForest + CatBoost).
    - input_df (pd.DataFrame): DataFrame with user-provided input features.
    - model_type (str): Type of base model to use ('rf' or 'cat').
    - confidence_level (float): Desired confidence level (e.g., 0.95).

    Returns:
    - str: Predicted price (formatted).
    - str: Lower bound of prediction interval (formatted).
    - str: Upper bound of prediction interval (formatted).
    - matplotlib.figure.Figure: SHAP waterfall plot.
    '''
    # Step 1: Prepare input data
    processed_data = prepare_input_data(input_df)
    
    # Step 2: Predict price
    prediction = model.predict(processed_data)[0]

    # Step 3: Estimate confidence interval using base model residual variance
    se = 0  # Default to zero if variance cannot be computed

    try:
        if model_type == 'rf':
            rf_model = model.named_estimators_['rf']
            y_train_pred = rf_model.predict(X_train)

            # Compute residuals from RF model
            residuals = y_train - y_train_pred
            variance = np.var(residuals)  # Use variance of residuals for standard error
            se = np.sqrt(variance)
        
        elif model_type == 'cat':
            cat_model = model.named_estimators_['cat']
            y_train_pred = cat_model.predict(X_train)

            # Compute residuals from CatBoost model
            residuals = y_train - y_train_pred
            variance = np.var(residuals)
            se = np.sqrt(variance)
        else:
            raise ValueError("Unsupported model type. Use 'rf' or 'cat'.")

    except Exception as e:
        print(f"Warning: Unable to compute confidence interval due to: {e}")

    # Step 4: Compute dynamic Z-score based on user-defined significance level
    if 0 < confidence_level < 1:
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)  # Works for any significance level!
    else:
        raise ValueError("Significance level must be between 0 and 1")

    # Only calculate interval if variance was computed successfully
    lower_bound = prediction - z_score * se
    upper_bound = prediction + z_score * se
    
    # Step 5: Generate SHAP waterfall plot
    shap_plot = plot_shap_waterfall(model, processed_data, model_type)

    return f"{prediction:,.2f}", f"{lower_bound:,.2f}", f"{upper_bound:,.2f}", shap_plot