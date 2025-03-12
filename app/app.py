from typing import Optional
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import numpy as np
from scipy.stats import norm
import os
import sys
from price_prediction import predict_house_price

# Load the trained model and other components
model = joblib.load("model_components/stacking_model.pkl")

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

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
    show_confidence: Optional[str] = Form(None),  # Capture toggle state
    alpha: float = Form(0.95),
):
    show_confidence_flag = show_confidence is not None  # Convert to boolean
    
    # Your existing validation
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
    
    # Get prediction results
    predicted_price, lower, upper, plot = predict_house_price(model, user_data, confidence_level=alpha)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "predicted_price": f"{float(predicted_price.replace(',', '')):,.2f}",
        "confidence_interval": f"[{float(lower.replace(',', '')):,.2f}, {float(upper.replace(',', '')):,.2f}]" if show_confidence_flag else None,
        "confidence_level": alpha if show_confidence_flag else None,
        "show_confidence": show_confidence_flag,  # Critical for template toggle
        # Preserve form values
        "area": area,
        "elevator": elevator,
        "rooms": rooms,
        "floor": floor,
        "region": region,
        "year_of_building": year_of_building,
        "alpha": alpha,
        "shap_plot": plot
    })