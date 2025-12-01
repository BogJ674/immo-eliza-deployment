from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv
import logging

from predict import predict, load_model_and_preprocessor

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title=os.getenv("API_TITLE", "Immo Eliza Price Prediction API"),
    version=os.getenv("API_VERSION", "1.0.0"),
    description=os.getenv("API_DESCRIPTION", "API for predicting real estate prices in Belgium")
)

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501,https://immo-eliza-deployment-y7rqynulwrysqimvfq2jfg.streamlit.app/").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
preprocessor = None
feature_names = None


class PropertyData(BaseModel):
    living_area: int = Field(..., description="Living area in square meters", ge=1)
    type_of_property: str = Field(..., description="Type of property: apartment, house, land, office, or garage")
    bedrooms: int = Field(..., description="Number of bedrooms", ge=0)
    postal_code: int = Field(..., description="Belgian postal code", ge=1000, le=9999)
    surface_of_good: Optional[int] = Field(None, description="Total surface of the property in square meters", ge=0)
    garden: Optional[bool] = Field(False, description="Has garden")
    garden_area: Optional[int] = Field(0, description="Garden area in square meters", ge=0)
    swimming_pool: Optional[bool] = Field(False, description="Has swimming pool")
    furnished: Optional[bool] = Field(False, description="Is furnished")
    openfire: Optional[bool] = Field(False, description="Has open fire")
    terrace: Optional[bool] = Field(False, description="Has terrace")
    number_of_facades: Optional[int] = Field(2, description="Number of facades", ge=1, le=4)
    construction_year: Optional[int] = Field(None, description="Year of construction", ge=1800, le=2025)
    state_of_building: Optional[str] = Field(None, description="State: 'to be done up', 'to restore', or 'to renovate'")
    kitchen: Optional[str] = Field(None, description="Kitchen: 'not installed', 'usa not installed', or 'installed'")


class PredictionResponse(BaseModel):
    prediction: Optional[float] = Field(None, description="Predicted price in EUR")
    status_code: int = Field(..., description="HTTP status code")
    message: Optional[str] = Field(None, description="Additional information or error message")


@app.on_event("startup")
async def startup_event():
    global model, preprocessor, feature_names

    model_path = os.getenv("MODEL_PATH", "models/xgboost_all_data.pkl")
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor_all_data.pkl")

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Loading preprocessor from {preprocessor_path}")

    try:
        if os.path.exists(preprocessor_path):
            model, preprocessor, feature_names = load_model_and_preprocessor(model_path, preprocessor_path)
            logger.info("Model and preprocessor loaded successfully")
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path}, will use model only")
            import joblib
            model = joblib.load(model_path)
            preprocessor = None
            feature_names = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    return {"status": "alive", "message": "Immo Eliza API is running"}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_price(data: PropertyData):
    try:
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        # Map state_of_building to state code
        state_map = {
            "to be done up": 0,
            "to restore": 1,
            "to renovate": 1,
            "good": 2,
            "as new": 3
        }
        state = state_map.get(data.state_of_building.lower(), -1) if data.state_of_building else -1

        # Map kitchen to has_equipped_kitchen
        has_equipped_kitchen = 1 if data.kitchen and data.kitchen.lower() == "installed" else -1

        # Build property dict with all required features (use -1 for missing/unknown values as in training data)
        property_dict = {
            # Basic property features
            "rooms": data.bedrooms,
            "area": data.living_area,
            "state": state,
            "facades_number": float(data.number_of_facades) if data.number_of_facades else -1.0,
            "is_furnished": int(data.furnished) if data.furnished else 0,
            "has_terrace": int(data.terrace) if data.terrace else 0,
            "has_garden": int(data.garden) if data.garden else 0,
            "has_swimming_pool": int(data.swimming_pool) if data.swimming_pool else -1,
            "has_equipped_kitchen": has_equipped_kitchen,
            "build_year": float(data.construction_year) if data.construction_year else 2000.0,
            "cellar": -1,  # Not provided in API
            "garage": -1,  # Not provided in API
            "kitchen_surface_house": -1.0,
            "bathrooms": float(max(1, data.bedrooms // 2)),
            "heating_type": -1,
            "terrace_surface_apartment": -1.0,
            "land_surface_house": -1.0,
            "sewer_connection": -1,
            "running_water": -1,
            "primary_energy_consumption": -1,
            "co2_house": -1,
            "certification_electrical_installation": -1,
            "preemption_right": -1,
            "flooding_area_type": -1,
            "leased": -1,
            "living_room_surface": -1,
            "attic_house": -1,
            "glazing_type": -1,
            "elevator": -1,
            "entry_phone_apartment": -1,
            "access_disabled": -1,
            "apartement_floor_apartment": -1,
            "number_floors_apartment": -1,
            "toilets": -1,
            "cadastral_income_house": -1,
            "postal_code": data.postal_code,
            "property_type": data.type_of_property.capitalize(),
            # These will be set to 0 by default, can be improved with postal code lookup
            "median_income_mun": 0,
            "median_income_arr": 0,
            "median_income_prv": 0,
            "median_price_house": 0,
            "median_price_apartment": 0,
        }

        predicted_price = predict(model, preprocessor, property_dict, feature_names)

        return PredictionResponse(
            prediction=float(predicted_price),
            status_code=status.HTTP_200_OK,
            message="Prediction successful"
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )


@app.get("/health", response_model=dict)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }
