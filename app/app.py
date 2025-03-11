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

from help_functions import predict_house_price

# Load the trained model and other components
model = joblib.load("model_components/stacking_model.pkl")

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

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
    alpha: float = Form(0.95),
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
    user_data = pd.DataFrame([user_data])
    predicted_price, lower, upper, plot = predict_house_price(model, user_data, confidence_level=alpha)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "predicted_price": f"{float(predicted_price.replace(',', '')):,.2f}",
        "confidence_interval": f"[{float(lower.replace(',', '')):,.2f}, {float(upper.replace(',', '')):,.2f}]",
        "confidence_level": alpha,
        "area": area,
        "elevator": elevator,
        "rooms": rooms,
        "floor": floor,
        "region": region,
        "year_of_building": year_of_building,
        "alpha": alpha,
    })