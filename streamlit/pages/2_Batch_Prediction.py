import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv
import io

load_dotenv()

API_URL = os.getenv("STREAMLIT_API_URL", "https://immo-eliza-deployment-kved.onrender.com/")

st.set_page_config(
    page_title="Batch Prediction",
    layout="wide"
)

# Include Bootstrap Icons CSS
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
""", unsafe_allow_html=True)

st.markdown('<h1><i class="bi bi-table"></i> Batch Property Prediction</h1>', unsafe_allow_html=True)
st.markdown("Upload a CSV file to predict prices for multiple properties at once")

# Check API health in sidebar
with st.sidebar:
    st.header("API Status")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            st.success("API Connected")
        else:
            st.error("API Connection Error")
    except:
        st.error("Cannot connect to API")

# Download sample CSV template
st.header("1. Download Sample Template (Optional)")
sample_data = {
    "postal_code": [2000, 1000, 9000],
    "living_area": [100, 150, 80],
    "type_of_property": ["apartment", "house", "apartment"],
    "bedrooms": [2, 3, 1],
    "construction_year": [2000, 1995, 2010],
    "state_of_building": ["good", "to renovate", "good"],
    "number_of_facades": [2, 4, 2],
    "kitchen": ["installed", "installed", "not installed"],
    "garden": [False, True, False],
    "terrace": [True, False, True],
    "swimming_pool": [False, True, False],
    "furnished": [False, False, True],
    "openfire": [False, True, False],
    "surface_of_good": [0, 200, 0],
    "garden_area": [0, 50, 0]
}
sample_df = pd.DataFrame(sample_data)

csv_buffer = io.StringIO()
sample_df.to_csv(csv_buffer, index=False)
csv_str = csv_buffer.getvalue()

st.download_button(
    label="Download Sample CSV Template",
    data=csv_str,
    file_name="property_template.csv",
    mime="text/csv",
    help="Download a sample CSV file with the correct format"
)

# File upload section
st.header("2. Upload Your CSV File")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file containing property data"
)

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        st.success(f"File uploaded successfully! Found {len(df)} properties.")

        # Display the uploaded data
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head(10), use_container_width=True)

        # Validate required columns
        required_columns = ['postal_code']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            # Process predictions button
            st.header("3. Generate Predictions")

            st.info(f"Ready to process {len(df)} properties")

            if st.button("Generate Predictions", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                predictions = []
                errors = []

                for idx, row in df.iterrows():
                    status_text.text(f"Processing property {idx + 1} of {len(df)}...")
                    progress_bar.progress((idx + 1) / len(df))

                    # Prepare property data
                    property_data = {
                        "postal_code": int(row['postal_code']),
                        "living_area": int(row['living_area']) if pd.notna(row.get('living_area')) else None,
                        "type_of_property": str(row['type_of_property']).lower() if pd.notna(row.get('type_of_property')) else None,
                        "bedrooms": int(row['bedrooms']) if pd.notna(row.get('bedrooms')) else None,
                        "construction_year": int(row['construction_year']) if pd.notna(row.get('construction_year')) else None,
                        "state_of_building": str(row['state_of_building']).lower() if pd.notna(row.get('state_of_building')) else None,
                        "number_of_facades": int(row['number_of_facades']) if pd.notna(row.get('number_of_facades')) else None,
                        "kitchen": str(row['kitchen']).lower() if pd.notna(row.get('kitchen')) else None,
                        "garden": bool(row['garden']) if pd.notna(row.get('garden')) else None,
                        "terrace": bool(row['terrace']) if pd.notna(row.get('terrace')) else None,
                        "swimming_pool": bool(row['swimming_pool']) if pd.notna(row.get('swimming_pool')) else None,
                        "furnished": bool(row['furnished']) if pd.notna(row.get('furnished')) else None,
                        "openfire": bool(row['openfire']) if pd.notna(row.get('openfire')) else None,
                        "surface_of_good": int(row['surface_of_good']) if pd.notna(row.get('surface_of_good')) and row.get('surface_of_good', 0) > 0 else None,
                        "garden_area": int(row['garden_area']) if pd.notna(row.get('garden_area')) else 0,
                    }

                    try:
                        # Make API request
                        response = requests.post(
                            f"{API_URL}/predict",
                            json=property_data,
                            timeout=10
                        )

                        if response.status_code == 200:
                            result = response.json()
                            prediction = result.get("prediction")
                            predictions.append(prediction)
                        else:
                            predictions.append(None)
                            errors.append(f"Row {idx + 1}: {response.json().get('detail', 'Unknown error')}")

                    except Exception as e:
                        predictions.append(None)
                        errors.append(f"Row {idx + 1}: {str(e)}")

                status_text.text("Processing complete!")
                progress_bar.empty()

                # Add predictions to dataframe
                df['predicted_price'] = predictions

                # Calculate price per sqm if living_area exists
                if 'living_area' in df.columns:
                    df['price_per_sqm'] = df.apply(
                        lambda row: row['predicted_price'] / row['living_area']
                        if pd.notna(row['predicted_price']) and pd.notna(row['living_area']) and row['living_area'] > 0
                        else None,
                        axis=1
                    )

                # Display results
                st.header("4. Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Properties", len(df))
                with col2:
                    successful = df['predicted_price'].notna().sum()
                    st.metric("Successful Predictions", successful)
                with col3:
                    failed = df['predicted_price'].isna().sum()
                    st.metric("Failed Predictions", failed)

                # Show errors if any
                if errors:
                    with st.expander(f"View Errors ({len(errors)})"):
                        for error in errors:
                            st.error(error)

                # Display results table
                st.subheader("Prediction Results")

                # Format the display dataframe
                display_df = df.copy()
                if 'predicted_price' in display_df.columns:
                    display_df['predicted_price'] = display_df['predicted_price'].apply(
                        lambda x: f"€{x:,.0f}" if pd.notna(x) else "Failed"
                    )
                if 'price_per_sqm' in display_df.columns:
                    display_df['price_per_sqm'] = display_df['price_per_sqm'].apply(
                        lambda x: f"€{x:,.0f}" if pd.notna(x) else "-"
                    )

                st.dataframe(display_df, use_container_width=True)

                # Download results
                st.subheader("Download Results")

                # Prepare CSV for download
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()

                st.download_button(
                    label="Download Results as CSV",
                    data=csv_str,
                    file_name="property_predictions.csv",
                    mime="text/csv",
                    help="Download the predictions as a CSV file"
                )

                # Summary statistics
                if successful > 0:
                    st.subheader("Summary Statistics")
                    col1, col2, col3 = st.columns(3)

                    valid_predictions = df[df['predicted_price'].notna()]['predicted_price']

                    with col1:
                        st.metric("Average Price", f"€{valid_predictions.mean():,.0f}")
                    with col2:
                        st.metric("Median Price", f"€{valid_predictions.median():,.0f}")
                    with col3:
                        st.metric("Price Range", f"€{valid_predictions.min():,.0f} - €{valid_predictions.max():,.0f}")

    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.info("Please make sure your CSV file is properly formatted.")

else:
    st.info("Please upload a CSV file to begin batch predictions.")
