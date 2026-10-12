import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src module is accessible
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.predict import RealEstatePredictor

# Page Configuration
st.set_page_config(
    page_title="AI Real Estate Valuation & Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 6px solid #2563EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .score-badge {
        font-size: 2.5rem;
        font-weight: 900;
        color: #1D4ED8;
    }
    .tier-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        background-color: #D1FAE5;
        color: #065F46;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'real_estate_gbr_pipeline.joblib')
    return RealEstatePredictor(model_path=model_path)

try:
    predictor = load_predictor()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load trained model pipeline: {e}. Please run model training first.")

# Title & Description
st.markdown('<div class="main-header">🏠 AI-Powered Real Estate Valuation & Investment Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Gradient Boosting Machine Learning Model with Explainable AI & Domain Valuation Scoring</div>', unsafe_allow_html=True)

if model_loaded:
    # Sidebar Input Form
    st.sidebar.header("📋 Property Specifications")
    
    cities = ['Chandigarh', 'Kolkata', 'Delhi', 'Ahmedabad', 'Pune', 'Hyderabad', 'Mumbai', 'Bangalore', 'Jaipur', 'Chennai']
    locality_types = ['Mid-Range', 'Premium', 'Affordable']
    property_types = ['Apartment', 'Villa', 'Builder Floor', 'Independent House']
    furnishing_options = ['Semi-Furnished', 'Unfurnished', 'Furnished']

    city = st.sidebar.selectbox("City", cities, index=2)
    locality = st.sidebar.selectbox("Locality Type", locality_types, index=0)
    prop_type = st.sidebar.selectbox("Property Type", property_types, index=0)
    furnishing = st.sidebar.selectbox("Furnishing Status", furnishing_options, index=0)

    st.sidebar.subheader("📐 Area & Layout")
    super_area = st.sidebar.number_input("Super Area (SqFt)", min_value=200, max_value=10000, value=1400, step=50)
    carpet_area = st.sidebar.number_input("Carpet Area (SqFt)", min_value=150, max_value=8000, value=1050, step=50)
    
    if carpet_area > super_area:
        st.sidebar.warning("⚠️ Warning: Carpet Area usually shouldn't exceed Super Area.")

    bhk = st.sidebar.slider("BHK (Bedrooms)", 1, 6, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 6, 3)
    
    st.sidebar.subheader("🏢 Building & Amenities")
    floor_num = st.sidebar.number_input("Floor Number", min_value=0, max_value=50, value=5)
    total_floors = st.sidebar.number_input("Total Floors in Building", min_value=1, max_value=50, value=15)
    age = st.sidebar.slider("Property Age (Years)", 0, 40, 5)

    if floor_num > total_floors:
        st.sidebar.warning("⚠️ Floor Number cannot be greater than Total Floors.")

    col_a1, col_a2 = st.sidebar.columns(2)
    with col_a1:
        parking = st.sidebar.checkbox("Parking Slot", value=True)
        lift = st.sidebar.checkbox("Lift / Elevator", value=True)
    with col_a2:
        gated = st.sidebar.checkbox("Gated Community", value=True)

    st.sidebar.subheader("📍 Transit & Location")
    dist_metro = st.sidebar.slider("Distance to Metro (km)", 0.1, 15.0, 2.5, step=0.1)
    dist_city = st.sidebar.slider("Distance to City Center (km)", 0.1, 30.0, 8.0, step=0.5)

    input_data = {
        'City': city,
        'Locality_Type': locality,
        'Property_Type': prop_type,
        'BHK': int(bhk),
        'Bathrooms': int(bathrooms),
        'Super_Area_SqFt': float(super_area),
        'Carpet_Area_SqFt': float(carpet_area),
        'Floor_Number': int(floor_num),
        'Total_Floors': int(total_floors),
        'Age_of_Property': int(age),
        'Furnishing_Status': furnishing,
        'Parking': 1 if parking else 0,
        'Lift_Available': 1 if lift else 0,
        'Gated_Community': 1 if gated else 0,
        'Distance_to_Metro_km': float(dist_metro),
        'Distance_to_City_Center_km': float(dist_city)
    }

    # Main Panel Tabs
    tab1, tab2, tab3 = st.tabs(["💰 Valuation & Intelligence", "📊 Model Insights & Analytics", "📁 Project Documentation"])

    with tab1:
        report = predictor.predict_with_intelligence(input_data)
        
        col1, col2, col3 = st.columns([1.5, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0; color:#1E3A8A;">Estimated Property Market Value</h4>
                <h1 style="margin:10px 0; font-size: 3rem; color:#1D4ED8;">₹ {report['Predicted_Price_Lakhs']} Lakhs</h1>
                <p style="margin:0; font-weight:600; color:#4B5563;">≈ ₹ {report['Predicted_Price_Crores']} Crores</p>
                <hr style="margin:12px 0;">
                <p style="margin:0; font-size:0.95rem;"><b>Valuation Range (84% Confidence):</b> ₹ {report['Valuation_Lower_Lakhs']} L – ₹ {report['Valuation_Upper_Lakhs']} L</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.metric("Price / SqFt (Super)", f"₹ {report['Price_Per_SqFt_Super']:,.0f}")
            st.metric("Price / SqFt (Carpet)", f"₹ {report['Price_Per_SqFt_Carpet']:,.0f}")
            
        with col3:
            st.markdown(f"""
            <div style="background:#F3F4F6; border-radius:12px; padding:18px; text-align:center;">
                <p style="margin:0; color:#4B5563; font-weight:700;">Valuation Score</p>
                <div class="score-badge">{report['Valuation_Score']} / 100</div>
                <div class="tier-badge">{report['Valuation_Tier']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("💡 Real Estate Intelligence Breakdown")
        
        col_dr1, col_dr2 = st.columns(2)
        
        with col_dr1:
            st.markdown("#### ✅ Key Positive Value Drivers")
            if report['Key_Value_Drivers']:
                for driver in report['Key_Value_Drivers']:
                    st.success(driver)
            else:
                st.info("Standard baseline property characteristics.")
                
        with col_dr2:
            st.markdown("#### ⚠️ Potential Discount Factors")
            if report['Discount_Factors']:
                for factor in report['Discount_Factors']:
                    st.warning(factor)
            else:
                st.success("No significant negative discount factors detected!")

        st.markdown("#### 📝 Executive Summary")
        st.info(report['Executive_Summary'])

    with tab2:
        st.subheader("📈 Model Diagnostics & Feature Importances")
        
        col_img1, col_img2 = st.columns(2)
        
        viz_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
        
        with col_img1:
            st.markdown("##### Top 15 Feature Importances (Gradient Boosting)")
            feat_img = os.path.join(viz_dir, 'feature_importance_top15.png')
            if os.path.exists(feat_img):
                st.image(feat_img, use_container_width=True)
            else:
                st.write("Feature importance visualization generating...")

        with col_img2:
            st.markdown("##### SHAP Summary Plot (Global Explainability)")
            shap_img = os.path.join(viz_dir, 'shap_summary_plot.png')
            if os.path.exists(shap_img):
                st.image(shap_img, use_container_width=True)
            else:
                st.write("SHAP visualization generating...")

        st.markdown("---")
        st.markdown("##### Actual vs Predicted Property Values")
        act_pred_img = os.path.join(viz_dir, 'actual_vs_predicted.png')
        if os.path.exists(act_pred_img):
            st.image(act_pred_img, use_container_width=True)

    with tab3:
        st.subheader("📑 Project Architecture & Model Specifications")
        st.markdown("""
        **Project Title:** AI-Powered Real Estate Valuation & Investment Intelligence System  
        **Primary Algorithm:** Tuned `GradientBoostingRegressor`  
        **Training Sample Size:** 80,000 Real Estate Records  
        **Evaluation Strategy:** 80-20 Leak-Free Train/Test Split + 5-Fold Cross Validation  
        **Preprocessing:** Custom Feature Engineering + OneHotEncoder + StandardScaler  
        
        ---
        ##### Key Performance Metrics:
        - **MAE:** Mean Absolute Error in ₹ Lakhs
        - **RMSE:** Root Mean Squared Error in ₹ Lakhs
        - **R²:** Coefficient of Determination
        """)

