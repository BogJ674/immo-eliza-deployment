import streamlit as st
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_URL = os.getenv("STREAMLIT_API_URL", "https://immo-eliza-deployment-kved.onrender.com/")

# Load logo and header image
logo_path = Path(__file__).parent / "assets" / "house_logo.svg"
header_image_path = Path(__file__).parent / "assets" / "file-6h8s.jpeg"

st.set_page_config(
    page_title="Immo Eliza - Price Prediction",
    page_icon=str(logo_path) if logo_path.exists() else "🏠",
    layout="wide"
)

# Include Bootstrap Icons CSS
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
""", unsafe_allow_html=True)

# Header with background image
if header_image_path.exists():
    import base64

    # Convert image to base64
    with open(header_image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode()

    # Create header with background image
    st.markdown(
        f"""
        <style>
        .header-container {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url('data:image/jpeg;base64,{img_data}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            padding: 60px 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header-title {{
            color: white !important;
            font-size: 3em;
            font-weight: bold;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
        }}
        .header-subtitle {{
            color: white !important;
            font-size: 1.3em;
            margin-top: 10px;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
        }}
        /* Override Streamlit's default header color */
        .header-container h1 {{
            color: white !important;
        }}
        </style>
        <div class="header-container">
            <h1 class="header-title">Immo Eliza - Real Estate Price Prediction</h1>
            <p class="header-subtitle">AI-powered property valuation for the Belgian market</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("Immo Eliza - Real Estate Price Prediction")
    st.markdown("AI-powered property valuation for the Belgian market")

# Sidebar with information
st.sidebar.header("About")
st.sidebar.info(
    "This application uses machine learning to predict real estate prices in Belgium. "
    "Choose between single property prediction or batch processing of multiple properties."
)

# Check API health
st.sidebar.markdown("---")
st.sidebar.header("API Status")
try:
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        st.sidebar.success("API Connected")
    else:
        st.sidebar.error("API Connection Error")
except:
    st.sidebar.error("Cannot connect to API")

# Main content - Welcome page
st.header("Welcome to Immo Eliza")

st.markdown("""
This application provides accurate real estate price predictions for properties in Belgium using advanced machine learning models.
""")

# Feature cards
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3><i class="bi bi-house-door"></i> Single Property Prediction</h3>', unsafe_allow_html=True)
    st.markdown("""
    Get an instant price prediction for a single property by entering its details.

    **Features:**
    - Interactive form with all property attributes
    - Instant predictions
    - Detailed price breakdown
    - Property summary

    Navigate to **Single Property** in the sidebar to get started.
    """)

with col2:
    st.markdown('<h3><i class="bi bi-table"></i> Batch Prediction</h3>', unsafe_allow_html=True)
    st.markdown("""
    Process multiple properties at once by uploading a CSV file.

    **Features:**
    - Upload CSV files with property data
    - Bulk predictions for multiple properties
    - Download results as CSV
    - Summary statistics

    Navigate to **Batch Prediction** in the sidebar to get started.
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Use the sidebar navigation to get started with predictions!</p>
</div>
""", unsafe_allow_html=True)
