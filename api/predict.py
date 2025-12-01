"""
Real Estate Price Prediction - Prediction Script

This script loads a trained model and makes predictions on new data.

Usage:
    python predict.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_model_and_preprocessor(model_path, preprocessor_path):
    """
    Load the trained model and preprocessor.

    Parameters:
    -----------
    model_path : str or Path
        Path to the saved model
    preprocessor_path : str or Path
        Path to the saved preprocessor

    Returns:
    --------
    tuple : (model, preprocessor, feature_names)
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading preprocessor from {preprocessor_path}...")
    pipeline_data = joblib.load(preprocessor_path)
    preprocessor = pipeline_data['preprocessor']
    feature_names = pipeline_data['feature_names']

    return model, preprocessor, feature_names


def create_sample_data():
    """
    Create sample property data for prediction.

    Returns:
    --------
    pd.DataFrame
        Sample property data
    """
    # Sample properties based on actual data structure
    samples = [
        {
            # Luxury apartment in Antwerp
            'postal_code': 2000,
            'rooms': 3,
            'area': 120,
            'state': 2,  # Good condition
            'facades_number': 2,
            'is_furnished': 0,
            'has_terrace': 1,
            'has_garden': 0,
            'has_swimming_pool': 0,
            'has_equipped_kitchen': 1,
            'build_year': 2015,
            'cellar': 1,
            'garage': 1,
            'bathrooms': 2,
            'heating_type': 5,  # Central heating
            'primary_energy_consumption': 100,
            'property_type': 'Apartment',
            'cluster': 1,
            'median_income_mun': 23.986,
            'median_income_arr': 29.71,
            'median_income_prv': 29.44
        },
        {
            # Family house in Brussels area
            'postal_code': 1050,
            'rooms': 4,
            'area': 150,
            'state': 2,
            'facades_number': 2,
            'is_furnished': 0,
            'has_terrace': 1,
            'has_garden': 1,
            'has_swimming_pool': 0,
            'has_equipped_kitchen': 1,
            'build_year': 2010,
            'cellar': 1,
            'garage': 1,
            'bathrooms': 2,
            'heating_type': 5,
            'primary_energy_consumption': 120,
            'property_type': 'House',
            'cluster': 1,
            'median_income_mun': 30.0,
            'median_income_arr': 28.5,
            'median_income_prv': 28.0
        },
        {
            # Small apartment in Ghent
            'postal_code': 9000,
            'rooms': 2,
            'area': 65,
            'state': 1,  # To renovate
            'facades_number': 2,
            'is_furnished': 0,
            'has_terrace': 0,
            'has_garden': 0,
            'has_swimming_pool': 0,
            'has_equipped_kitchen': 0,
            'build_year': 1980,
            'cellar': 0,
            'garage': 0,
            'bathrooms': 1,
            'heating_type': 3,
            'primary_energy_consumption': 200,
            'property_type': 'Apartment',
            'cluster': 0,
            'median_income_mun': 26.5,
            'median_income_arr': 27.0,
            'median_income_prv': 27.5
        }
    ]

    return pd.DataFrame(samples)


def prepare_prediction_data(df, feature_names):
    """
    Prepare data for prediction by ensuring all required features exist.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_names : dict
        Dictionary with 'numeric' and 'categorical' feature lists

    Returns:
    --------
    pd.DataFrame
        Prepared dataframe with all required features
    """
    numeric_features = feature_names['numeric']
    categorical_features = feature_names['categorical']

    all_features = numeric_features + categorical_features

    # Add missing features with default values
    for feature in all_features:
        if feature not in df.columns:
            if feature in numeric_features:
                df[feature] = -1  # Default for numeric
            else:
                df[feature] = 'Unknown'  # Default for categorical

    # Ensure correct column order
    df = df[all_features]

    return df


def predict_prices(model, preprocessor, data):
    """
    Make price predictions for the given data.

    Parameters:
    -----------
    model : sklearn model
        Trained model
    preprocessor : sklearn transformer
        Fitted preprocessor
    data : pd.DataFrame
        Property data

    Returns:
    --------
    np.array
        Predicted prices
    """
    # Preprocess data
    data_processed = preprocessor.transform(data)

    # Make predictions
    predictions = model.predict(data_processed)

    return predictions


def main():
    """Main prediction pipeline."""

    print("=" * 80)
    print("REAL ESTATE PRICE PREDICTION")
    print("=" * 80)

    # Determine which model directory to use based on available models
    # Priority: no_split > split_cluster > split_prop_type > split_both
    possible_strategies = ['no_split', 'split_cluster', 'split_prop_type', 'split_both']

    model_path = None
    preprocessor_path = None
    strategy_used = None

    for strategy in possible_strategies:
        # Try to find models in this strategy directory
        strategy_dir = Path("models") / strategy

        # Look for any model file in the directory
        if strategy_dir.exists():
            model_files = list(strategy_dir.glob("*_all_data.pkl")) or list(strategy_dir.glob("*.pkl"))
            preprocessor_files = list(strategy_dir.glob("preprocessor*.pkl"))

            if model_files and preprocessor_files:
                # Use the first model found (preferably the all_data one)
                all_data_models = [f for f in model_files if 'all_data' in f.name]
                model_path = all_data_models[0] if all_data_models else model_files[0]
                preprocessor_path = preprocessor_files[0]
                strategy_used = strategy
                break

    # Check if files exist
    if not model_path or not model_path.exists():
        print(f"\n❌ Error: No trained models found in models/ subdirectories")
        print("Please run train_all_models.py first to train the models.")
        print("\nLooking for models in:")
        for strategy in possible_strategies:
            print(f"  - models/{strategy}/")
        return

    if not preprocessor_path or not preprocessor_path.exists():
        print(f"\n❌ Error: Preprocessor file not found")
        print("Please run train_all_models.py first to create the preprocessor.")
        return

    print(f"\n📁 Using models from: models/{strategy_used}/")
    print(f"   Model: {model_path.name}")
    print(f"   Preprocessor: {preprocessor_path.name}")

    # Load model and preprocessor
    print("\n[1/4] Loading model and preprocessor...")
    model, preprocessor, feature_names = load_model_and_preprocessor(
        model_path, preprocessor_path
    )
    print("   ✓ Model and preprocessor loaded successfully")

    # Create or load sample data
    print("\n[2/4] Loading property data...")
    df = create_sample_data()
    print(f"   ✓ Loaded {len(df)} properties for prediction")

    # Prepare data
    print("\n[3/4] Preparing data...")
    df_prepared = prepare_prediction_data(df, feature_names)
    print(f"   ✓ Data prepared with {len(df_prepared.columns)} features")

    # Make predictions
    print("\n[4/4] Making predictions...")
    predictions = predict_prices(model, preprocessor, df_prepared)
    print("   ✓ Predictions completed")

    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)

    for i, (idx, row) in enumerate(df.iterrows()):
        print(f"\nProperty {i+1}:")
        print(f"  Location:      Postal Code {int(row['postal_code'])}")
        print(f"  Type:          {row['property_type']}")
        print(f"  Size:          {int(row['area'])} m²")
        print(f"  Rooms:         {int(row['rooms'])}")
        print(f"  Condition:     {'To renovate' if row['state'] == 1 else 'Good' if row['state'] == 2 else 'As new'}")
        print(f"  Build Year:    {int(row['build_year'])}")
        print(f"  Garden:        {'Yes' if row['has_garden'] else 'No'}")
        print(f"  Garage:        {'Yes' if row['garage'] else 'No'}")
        print(f"\n  → Predicted Price: €{predictions[i]:,.2f}")
        print(f"  → Price per m²:    €{predictions[i]/row['area']:,.2f}")
        print("  " + "-" * 60)

    print("\n" + "=" * 80)
    print("✓ PREDICTIONS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
