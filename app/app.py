from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import numpy as np
from scipy.stats import norm, boxcox
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load the trained model and other components
model = joblib.load("../model_components/random_forest_model.pkl")
label_encoder_floor = joblib.load("../model_components/label_encoder_floor.pkl")
label_encoder_region = joblib.load("../model_components/label_encoder_region.pkl")
lambda_area = joblib.load("../model_components/lambda_area.pkl")
region_weights = joblib.load("../model_components/region_weights.pkl")
floor_weights = joblib.load("../model_components/floor_weights.pkl")
room_weights = joblib.load("../model_components/room_weights.pkl")
expected_features = joblib.load("../model_components/expected_features.pkl")

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

def predict_house_price(
    user_data, 
    model: RandomForestRegressor, 
    label_encoder_floor: LabelEncoder, 
    label_encoder_region: LabelEncoder, 
    lambda_area: float, 
    region_weights: dict, 
    floor_weights: dict, 
    room_weights: dict, 
    expected_features: list, 
    significance_level=0.05
):
    # Convert user input to DataFrame
    data = pd.DataFrame([user_data])

    # Merge floors 5 and above into "5+ piętro"
    high_floors = ['5 piętro', '6 piętro', '7 piętro', '8 piętro', '9 piętro', '10 piętro', '10+ piętro']
    data['Floor'] = data['Floor'].replace(high_floors, '5+ piętro')

    # Handle regions - replace low-frequency ones with "Other"
    if user_data['Region'] not in region_weights:
        data['Region'] = 'Other'

    # Apply Box-Cox Transformation for Area
    data['Area'] = boxcox(data['Area'], lambda_area)

    # Label encoding for categorical features
    data['Floor'] = label_encoder_floor.transform(data['Floor'])
    data['Region'] = label_encoder_region.transform(data['Region'])

    # Convert Elevator to binary (if not already)
    data['Elevator'] = data['Elevator'].apply(lambda x: 1 if x in [1, 'Yes', 'yes', 'y', 'true', True] else 0)

    # Ensure correct feature order and drop unexpected features
    missing_features = [feature for feature in expected_features if feature not in data.columns]
    extra_features = [feature for feature in data.columns if feature not in expected_features]

    if missing_features:
        raise ValueError(f"🚨 Missing required features: {missing_features}")

    if extra_features:
        print(f"⚠️ Warning: Ignoring unexpected features: {extra_features}")

    # Ensure correct column order & drop any extra columns
    data = data[expected_features]

    # Ensure correct data types
    data = data.astype({
        'Area': 'float64',
        'Elevator': 'float64',
        'Year': 'int32',
        'Rooms': 'float64',
        'Floor': 'int32',
        'Region': 'int32'
    })

    # Check feature names and order before prediction
    print("✅ Features expected by model:", list(model.feature_names_in_))
    print("✅ Features provided for prediction:", list(data.columns))

    # Final validation before prediction
    if not np.array_equal(model.feature_names_in_, data.columns.to_numpy()):
        raise ValueError("🚨 Feature names or order do not match what was used during model training.")

    # Predict house price
    predicted_price = model.predict(data)[0]

    # Get predictions from all individual trees in the Random Forest
    tree_predictions = np.array([tree.predict(data)[0] for tree in model.estimators_])

    # Compute the standard deviation of tree predictions
    std_dev = np.std(tree_predictions)

    # Compute confidence interval based on standard normal distribution
    z_score = norm.ppf(1 - significance_level / 2)  # Two-tailed z-score for given significance level
    margin_of_error = z_score * std_dev

    # Calculate confidence interval
    lower_bound = max(0, predicted_price - margin_of_error)  # Ensuring price is not negative
    upper_bound = predicted_price + margin_of_error

    return predicted_price, (lower_bound, upper_bound)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    area: float = Form(...),
    elevator: int = Form(...),
    rooms: int = Form(...),
    floor: str = Form(...),
    region: str = Form(...),
    year_of_building: int = Form(...),
    alpha: float = Form(0.05),
):
    if elevator not in [0, 1]:
        raise HTTPException(status_code=400, detail="Invalid value for elevator. Must be 0 or 1.")

    user_data = {
        "Rooms": rooms,
        "Area": area,
        "Floor": floor,
        "Region": region,
        "Elevator": elevator,
        "Year": year_of_building
    }

    predicted_price, (lower, upper) = predict_house_price(
        user_data, model, label_encoder_floor, label_encoder_region, lambda_area, 
        region_weights, floor_weights, room_weights, expected_features, significance_level=alpha
    )
    confidence_level = (1 - alpha) * 100

    return templates.TemplateResponse("index.html", {
        "request": request,
        "predicted_price": f"{predicted_price:,.2f}",
        "confidence_interval": f"[{lower:,.2f}, {upper:,.2f}]",
        "confidence_level": confidence_level,
        "area": area,
        "elevator": elevator,
        "rooms": rooms,
        "floor": floor,
        "region": region,
        "year_of_building": year_of_building,
        "alpha": alpha,
    })