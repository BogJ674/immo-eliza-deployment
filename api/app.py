from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal, Any
import os
from dotenv import load_dotenv
import logging
import json
from pathlib import Path

from predict import predict, load_model_and_preprocessor, calculate_values_from_postal_code

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
feature_defaults = None


class PropertyData(BaseModel):
    living_area: Optional[int] = Field(None, description="Living area in square meters", ge=1)
    type_of_property: Optional[str] = Field(None, description="Type of property: apartment, house, land, office, or garage")
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms", ge=0)
    postal_code: int = Field(..., description="Belgian postal code", ge=1000, le=9999)
    surface_of_good: Optional[int] = Field(None, description="Total surface of the property in square meters", ge=0)
    garden: Optional[bool] = Field(None, description="Has garden")
    garden_area: Optional[int] = Field(None, description="Garden area in square meters", ge=0)
    swimming_pool: Optional[bool] = Field(None, description="Has swimming pool")
    furnished: Optional[bool] = Field(None, description="Is furnished")
    openfire: Optional[bool] = Field(None, description="Has open fire")
    terrace: Optional[bool] = Field(None, description="Has terrace")
    number_of_facades: Optional[int] = Field(None, description="Number of facades", ge=1, le=4)
    construction_year: Optional[int] = Field(None, description="Year of construction", ge=1800, le=2025)
    state_of_building: Optional[str] = Field(None, description="State: 'to be done up', 'to restore', or 'to renovate'")
    kitchen: Optional[str] = Field(None, description="Kitchen: 'not installed', 'usa not installed', or 'installed'")


class PredictionResponse(BaseModel):
    prediction: Optional[float] = Field(None, description="Predicted price in EUR")
    status_code: int = Field(..., description="HTTP status code")
    message: Optional[str] = Field(None, description="Additional information or error message")


@app.on_event("startup")
async def startup_event():
    global model, preprocessor, feature_names, feature_defaults

    # Get paths from env or use defaults (relative to api directory)
    model_path = os.getenv("MODEL_PATH", "models/xgboost_all_data.pkl")
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor_all_data.pkl")
    defaults_path = os.getenv("FEATURE_DEFAULTS_PATH", "data/feature_defaults.json")

    # If paths start with "api/", strip it since we're running from the api directory
    if model_path.startswith("api/"):
        model_path = model_path[4:]
    if preprocessor_path.startswith("api/"):
        preprocessor_path = preprocessor_path[4:]
    if defaults_path.startswith("api/"):
        defaults_path = defaults_path[4:]

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Loading preprocessor from {preprocessor_path}")
    logger.info(f"Loading feature defaults from {defaults_path}")

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

        # Load feature defaults
        if os.path.exists(defaults_path):
            with open(defaults_path, 'r') as f:
                feature_defaults = json.load(f)
            logger.info(f"Feature defaults loaded successfully ({len(feature_defaults)} features)")
        else:
            logger.warning(f"Feature defaults not found at {defaults_path}, will use hardcoded defaults")
            feature_defaults = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def get_default_value(feature_name: str, fallback_value: Any = -1) -> Any:
    """
    Get the default value for a feature from the feature_defaults dictionary.

    Parameters:
    -----------
    feature_name : str
        Name of the feature
    fallback_value : Any
        Value to return if feature_defaults is not loaded

    Returns:
    --------
    Any : Default value for the feature
    """
    if feature_defaults is not None and feature_name in feature_defaults:
        return feature_defaults[feature_name]
    return fallback_value


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

        calculated_values = calculate_values_from_postal_code(data.postal_code)

        # Build property dict with all required features
        # Use provided values if available, otherwise use average values from training data
        property_dict = {
            # Basic property features
            "rooms": data.bedrooms if data.bedrooms is not None else get_default_value("rooms"),
            "area": data.living_area if data.living_area is not None else get_default_value("area"),
            "state": state if state != -1 else get_default_value("state"),
            "facades_number": float(data.number_of_facades) if data.number_of_facades is not None else get_default_value("facades_number"),
            "is_furnished": int(data.furnished) if data.furnished is not None else get_default_value("is_furnished", 0),
            "has_terrace": int(data.terrace) if data.terrace is not None else get_default_value("has_terrace", 0),
            "has_garden": int(data.garden) if data.garden is not None else get_default_value("has_garden", 0),
            "has_swimming_pool": int(data.swimming_pool) if data.swimming_pool is not None else get_default_value("has_swimming_pool", 0),
            "has_equipped_kitchen": has_equipped_kitchen if has_equipped_kitchen != -1 else get_default_value("has_equipped_kitchen"),
            "build_year": float(data.construction_year) if data.construction_year is not None else get_default_value("build_year", 2000.0),
            "cellar": get_default_value("cellar"),
            "garage": get_default_value("garage"),
            "kitchen_surface_house": get_default_value("kitchen_surface_house"),
            "bathrooms": float(max(1, data.bedrooms // 2)) if data.bedrooms is not None else get_default_value("bathrooms"),
            "heating_type": get_default_value("heating_type"),
            "terrace_surface_apartment": get_default_value("terrace_surface_apartment"),
            "land_surface_house": get_default_value("land_surface_house"),
            "sewer_connection": get_default_value("sewer_connection"),
            "running_water": get_default_value("running_water"),
            "primary_energy_consumption": get_default_value("primary_energy_consumption"),
            "co2_house": get_default_value("co2_house"),
            "certification_electrical_installation": get_default_value("certification_electrical_installation"),
            "preemption_right": get_default_value("preemption_right"),
            "flooding_area_type": get_default_value("flooding_area_type"),
            "leased": get_default_value("leased"),
            "living_room_surface": get_default_value("living_room_surface"),
            "attic_house": get_default_value("attic_house"),
            "glazing_type": get_default_value("glazing_type"),
            "elevator": get_default_value("elevator"),
            "entry_phone_apartment": get_default_value("entry_phone_apartment"),
            "access_disabled": get_default_value("access_disabled"),
            "apartement_floor_apartment": get_default_value("apartement_floor_apartment"),
            "number_floors_apartment": get_default_value("number_floors_apartment"),
            "toilets": get_default_value("toilets"),
            "cadastral_income_house": get_default_value("cadastral_income_house"),
            "postal_code": data.postal_code,
            "property_type": data.type_of_property.capitalize() if data.type_of_property is not None else get_default_value("property_type", "Apartment"),
            "median_income_mun": calculated_values["median_income_mun"],
            "median_income_arr": calculated_values["median_income_arr"],
            "median_income_prv": calculated_values["median_income_prv"],
            "median_price_house": calculated_values["median_price_house"],
            "median_price_apartment": calculated_values["median_price_apartment"],
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
