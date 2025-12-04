"""
Script to calculate average values for all features from training data.
These averages will be used as defaults for missing values in the API.
"""

import pandas as pd
import json
from pathlib import Path

def calculate_feature_defaults(csv_path):
    print(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows")

    # For features that have -1 as missing indicator, we'll calculate mean excluding -1
    # For boolean features (0/1), we'll use mode
    # For categorical features, we'll use mode

    feature_defaults = {}

    # Numeric features - calculate mean excluding -1 values
    numeric_features = [
        'rooms', 'area', 'state', 'facades_number', 'build_year',
        'bathrooms', 'heating_type', 'primary_energy_consumption',
        'kitchen_surface_house', 'terrace_surface_apartment', 'land_surface_house',
        'co2_house', 'living_room_surface', 'cadastral_income_house',
        'apartement_floor_apartment', 'number_floors_apartment', 'toilets',
        'glazing_type', 'flooding_area_type'
    ]

    for feature in numeric_features:
        if feature in df.columns:
            # Filter out -1 values before calculating mean
            valid_values = df[feature][df[feature] != -1]
            if len(valid_values) > 0:
                feature_defaults[feature] = float(valid_values.mean())
            else:
                feature_defaults[feature] = -1.0
            print(f"  {feature}: {feature_defaults[feature]:.2f}")

    # Boolean features (0/1 or -1 for missing) - use mode of non-missing values
    boolean_features = [
        'is_furnished', 'has_terrace', 'has_garden', 'has_swimming_pool',
        'has_equipped_kitchen', 'cellar', 'garage', 'sewer_connection',
        'running_water', 'certification_electrical_installation',
        'preemption_right', 'leased', 'attic_house', 'elevator',
        'entry_phone_apartment', 'access_disabled'
    ]

    for feature in boolean_features:
        if feature in df.columns:
            # Filter out -1 values before calculating mode
            valid_values = df[feature][df[feature] != -1]
            if len(valid_values) > 0:
                feature_defaults[feature] = int(valid_values.mode()[0])
            else:
                feature_defaults[feature] = -1
            print(f"  {feature}: {feature_defaults[feature]}")

    # Property type - use mode
    if 'property_type' in df.columns:
        feature_defaults['property_type'] = df['property_type'].mode()[0]
        print(f"  property_type: {feature_defaults['property_type']}")

    # Postal code median values - calculate overall means
    location_features = [
        'median_income_mun', 'median_income_arr', 'median_income_prv',
        'median_price_house', 'median_price_apartment'
    ]

    for feature in location_features:
        if feature in df.columns:
            feature_defaults[feature] = float(df[feature].mean())
            print(f"  {feature}: {feature_defaults[feature]:.2f}")

    return feature_defaults


def main():
    """Main function to generate feature defaults JSON file."""

    print("=" * 80)
    print("GENERATING FEATURE DEFAULTS FROM TRAINING DATA")
    print("=" * 80 + "\n")

    # Path to training data
    csv_path = Path(__file__).parent / "data" / "cleaned_dataset_v6.csv"

    if not csv_path.exists():
        print(f"Error: Training data not found at {csv_path}")
        return

    # Calculate defaults
    feature_defaults = calculate_feature_defaults(csv_path)

    # Save to JSON file
    output_path = Path(__file__).parent / "data" / "feature_defaults.json"

    print(f"\nSaving feature defaults to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(feature_defaults, f, indent=2)

    print("\n" + "=" * 80)
    print("✓ FEATURE DEFAULTS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nGenerated {len(feature_defaults)} feature defaults")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
