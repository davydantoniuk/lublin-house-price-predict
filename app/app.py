from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import numpy as np
from scipy.stats import norm
import os
import sys

# Get the absolute path of the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
# Move one level up
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)  # Add to Python's search path

# Change the working directory
os.chdir(project_root)

from help_functions import prepare_input_data

# Load the trained model and other components
model = joblib.load("model_components/stacking_model.pkl")
area_lambda = joblib.load("model_components/boxcox_lambda.pkl")
year_lower, year_upper = joblib.load("model_components/winsorization_limits.pkl")
X_train, y_train, X_test, y_test = joblib.load("model_components/data_split.pkl")

# Precompute residuals and standard errors for base models
rf_model = model.named_estimators_['rf']
cat_model = model.named_estimators_['cat']

# Compute residuals for RandomForest
y_train_pred_rf = rf_model.predict(X_train)
residuals_rf = y_train - y_train_pred_rf
variance_rf = np.var(residuals_rf)
se_rf = np.sqrt(variance_rf)

# Compute residuals for CatBoost
y_train_pred_cat = cat_model.predict(X_train)
residuals_cat = y_train - y_train_pred_cat
variance_cat = np.var(residuals_cat)
se_cat = np.sqrt(variance_cat)

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

def predict_house_price(user_data, model, significance_level=0.05):
    # Convert user input to DataFrame
    data = pd.DataFrame([user_data])
    
    # Prepare the data using the imported function
    processed_data = prepare_input_data(data)
    
    # Predict using the stacking model
    predicted_price = model.predict(processed_data)[0]
    
    # Compute confidence interval using precomputed se_rf
    z_score = norm.ppf(1 - significance_level / 2)
    margin_of_error = z_score * se_rf  # Using precomputed se_rf
    
    lower_bound = max(0, predicted_price - margin_of_error)
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

    predicted_price, (lower, upper) = predict_house_price(user_data, model, significance_level=alpha)
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