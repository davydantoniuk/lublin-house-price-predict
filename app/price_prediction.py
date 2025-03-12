import pandas as pd  
import numpy as np  
import joblib  
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap  
import matplotlib.pyplot as plt  
import warnings  

warnings.filterwarnings("ignore")

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