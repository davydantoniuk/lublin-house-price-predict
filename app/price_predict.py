from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("../model_components/final_catboost_model.joblib")
scaler = joblib.load("../model_components/scaler.joblib")  
X_columns = joblib.load("../model_components/X_columns.joblib")  
numerical_columns = ["Area", "Rooms"]  

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Define the prediction function
def predict_house_price(model, area, elevator, rooms, floor, region, year_of_building):
    if floor == 0:
        floor_category = "parter"
    elif floor <= 10:
        floor_category = f"{floor} piętro"
    else:
        floor_category = "10 piętro"

    if year_of_building < 1890:
        year_interval_category = "<1890"
    elif 1890 <= year_of_building <= 1920:
        year_interval_category = "1890-1920"
    elif 1921 <= year_of_building <= 1950:
        year_interval_category = "1921-1950"
    elif 1951 <= year_of_building <= 1980:
        year_interval_category = "1951-1980"
    elif 1981 <= year_of_building <= 2001:
        year_interval_category = "1981-2001"
    elif 2002 <= year_of_building <= 2016:
        year_interval_category = "2001-2016"
    else:
        year_interval_category = "2016<"

    # Initialize an empty DataFrame with the structure of X
    input_data = pd.DataFrame(columns=X_columns)
    input_data.loc[0, "Area"] = area
    input_data.loc[0, "Elevator"] = elevator
    input_data.loc[0, "Rooms"] = rooms
    input_data.loc[0, "Floor"] = floor
    input_data.loc[0, "Year_of_building"] = year_of_building

    if f"Floor_{floor_category}" in input_data.columns:
        input_data.loc[0, f"Floor_{floor_category}"] = 1
    if f"Region_{region}" in input_data.columns:
        input_data.loc[0, f"Region_{region}"] = 1
    if f"Year_interval_{year_interval_category}" in input_data.columns:
        input_data.loc[0, f"Year_interval_{year_interval_category}"] = 1

    input_data = input_data.fillna(0)
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Route for rendering the HTML form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route for handling the form submission and returning the prediction
@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    area: float = Form(...),
    elevator: int = Form(...),
    rooms: int = Form(...),
    floor: int = Form(...),
    region: str = Form(...),
    year_of_building: int = Form(...)
):
    if elevator not in [0, 1]:
        raise HTTPException(status_code=400, detail="Invalid value for elevator. Must be 0 or 1.")
    predicted_price = predict_house_price(model, area, elevator, rooms, floor, region, year_of_building)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "predicted_price": f"{predicted_price:,.2f}",
        "area": area,
        "elevator": elevator,
        "rooms": rooms,
        "floor": floor,
        "region": region,
        "year_of_building": year_of_building
    })