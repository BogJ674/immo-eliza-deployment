import streamlit as st
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_URL = os.getenv("STREAMLIT_API_URL", "https://immo-eliza-deployment-kved.onrender.com/")

st.set_page_config(
    page_title="Single Property Prediction",
    layout="wide"
)

# Include Bootstrap Icons CSS
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
""", unsafe_allow_html=True)

st.markdown('<h1><i class="bi bi-house-door"></i> Single Property Prediction</h1>', unsafe_allow_html=True)
st.markdown("Get an instant price prediction for a single property")

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

# Main form
st.header("Property Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Information")

    property_type = st.selectbox(
        "Property Type",
        ["Apartment", "House"],
        help="Select the type of property"
    )

    living_area = st.number_input(
        "Living Area (m²)",
        min_value=1,
        max_value=1000,
        value=100,
        help="Total living area in square meters"
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=0,
        max_value=10,
        value=2,
        help="Number of bedrooms"
    )

    postal_code = st.number_input(
        "Postal Code",
        min_value=1000,
        max_value=9999,
        value=2000,
        help="Belgian postal code"
    )

    construction_year = st.number_input(
        "Construction Year",
        min_value=1800,
        max_value=2025,
        value=2000,
        help="Year when the property was built"
    )

with col2:
    st.subheader("Additional Features")

    state_of_building = st.selectbox(
        "State of Building",
        ["Good", "To renovate", "To restore", "To be done up"],
        index=None,
        placeholder="Select state...",
    )

    number_of_facades = st.slider(
        "Number of Facades",
        min_value=1,
        max_value=4,
        value=2,
        help="Number of building facades"
    )

    kitchen = st.selectbox(
        "Kitchen",
        ["Installed", "Not installed", "USA not installed"],
        help="Kitchen installation status"
    )

    col2a, col2b = st.columns(2)

    with col2a:
        garden = st.checkbox("Garden", value=False)
        terrace = st.checkbox("Terrace", value=False)
        furnished = st.checkbox("Furnished", value=False)

    with col2b:
        swimming_pool = st.checkbox("Swimming Pool", value=False)
        openfire = st.checkbox("Open Fire", value=False)

# Optional fields in expander
with st.expander("Advanced Options"):
    surface_of_good = st.number_input(
        "Total Surface (m²) - Optional",
        min_value=0,
        max_value=5000,
        value=0,
        help="Total surface area of the property"
    )

    garden_area = st.number_input(
        "Garden Area (m²) - Optional",
        min_value=0,
        max_value=5000,
        value=0,
        help="Garden area in square meters"
    )

# Predict button
if st.button("Predict Price", type="primary", use_container_width=True):
    with st.spinner("Calculating prediction..."):
        # Prepare data
        property_data = {
            "living_area": living_area,
            "type_of_property": property_type.lower(),
            "bedrooms": bedrooms,
            "postal_code": postal_code,
            "surface_of_good": surface_of_good if surface_of_good > 0 else None,
            "garden": garden,
            "garden_area": garden_area if garden_area > 0 else 0,
            "swimming_pool": swimming_pool,
            "furnished": furnished,
            "openfire": openfire,
            "terrace": terrace,
            "number_of_facades": number_of_facades,
            "construction_year": construction_year,
            "state_of_building": state_of_building.lower() if state_of_building else None,
            "kitchen": kitchen.lower()
        }

        try:
            # Make API request
            response = requests.post(
                f"{API_URL}/predict",
                json=property_data
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")

                if prediction:
                    st.success("Prediction Complete!")

                    # Display result
                    st.markdown("---")
                    col_res1, col_res2, col_res3 = st.columns(3)

                    with col_res1:
                        st.metric(
                            label="Predicted Price",
                            value=f"€{prediction:,.0f}"
                        )

                    with col_res2:
                        price_per_sqm = prediction / living_area
                        st.metric(
                            label="Price per m²",
                            value=f"€{price_per_sqm:,.0f}"
                        )

                    with col_res3:
                        st.metric(
                            label="Property Type",
                            value=property_type
                        )

                    # Display property summary
                    st.markdown("---")
                    st.subheader("Property Summary")
                    summary_col1, summary_col2 = st.columns(2)

                    with summary_col1:
                        st.write(f"**Location:** Postal Code {postal_code}")
                        st.write(f"**Type:** {property_type}")
                        st.write(f"**Living Area:** {living_area} m²")
                        st.write(f"**Bedrooms:** {bedrooms}")

                    with summary_col2:
                        st.write(f"**Built:** {construction_year}")
                        st.write(f"**Condition:** {state_of_building}")
                        st.write(f"**Facades:** {number_of_facades}")
                        st.write(f"**Kitchen:** {kitchen}")

                    features = []
                    if garden:
                        features.append("Garden")
                    if terrace:
                        features.append("Terrace")
                    if swimming_pool:
                        features.append("Swimming Pool")
                    if furnished:
                        features.append("Furnished")
                    if openfire:
                        features.append("Open Fire")

                    if features:
                        st.write("**Features:**", " | ".join(features))
                else:
                    st.error("No prediction returned from the API")
            else:
                st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to the API at {API_URL}. Please make sure the API is running.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
