# AI-Powered Real Estate Valuation & Investment Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-red.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end Machine Learning and Decision Intelligence platform engineered to predict real estate property values, provide transparent prediction explanations (Explainable AI via SHAP), benchmark properties against macro-urban housing markets, and generate investment-grade decision recommendations.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [Key Features](#4-key-features)
5. [System Architecture](#5-system-architecture)
6. [Machine Learning Methodology](#6-machine-learning-methodology)
7. [Dataset Description & Features](#7-dataset-description--features)
8. [Data Preprocessing & Leakage Prevention](#8-data-preprocessing--leakage-prevention)
9. [Feature Engineering Strategy](#9-feature-engineering-strategy)
10. [Model Selection & Training](#10-model-selection--training)
11. [Hyperparameter Tuning & Validation](#11-hyperparameter-tuning--validation)
12. [Model Performance & Comparison Table](#12-model-performance--comparison-table)
13. [Explainable AI & SHAP Interpretation](#13-explainable-ai--shap-interpretation)
14. [Real Estate Domain Intelligence Engine](#14-real-estate-domain-intelligence-engine)
15. [Backend Architecture & FastAPI Endpoints](#15-backend-architecture--fastapi-endpoints)
16. [Frontend User Interface Structure](#16-frontend-user-interface-structure)
17. [Visualizations & Diagnostic Plots](#17-visualizations--diagnostic-plots)
18. [Technology Stack](#18-technology-stack)
19. [Project Directory Structure](#19-project-directory-structure)
20. [Installation & Setup Guide](#20-installation--setup-guide)
21. [Usage Guide & API Testing](#21-usage-guide--api-testing)
22. [Key Insights & Business Takeaways](#22-key-insights--business-takeaways)

---

## 1. Project Overview

The **AI-Powered Real Estate Valuation & Investment Intelligence System** bridges the gap between raw statistical regression and commercial real estate underwriting. While traditional Automated Valuation Models (AVMs) act as black-box calculators outputting static point estimates, this platform delivers a transparent, interpretable, and commercially grounded appraisal experience. 

Built with a high-performance **FastAPI** backend and an intuitive, modern responsive **Web Interface**, the system evaluates properties across 10 tier-1 Indian metropolitan housing markets, generates 95% valuation confidence intervals, dissects marginal feature impacts using **SHAP (SHapley Additive exPlanations)**, and scores asset viability through an algorithmic **Valuation Quality Index (0–100)**.

---

## 2. Problem Statement

Residential real estate valuation in emerging metropolitan markets faces critical challenges:
- **Spatial Heterogeneity & Non-Linearity:** Property values are dictated by multi-layered interactions between location premiums, transit access, spatial density, and amenity availability that simple linear models fail to capture.
- **The "Black-Box" Adoption Barrier:** Homebuyers, mortgage underwriters, and institutional investors distrust algorithmic estimates when there is no auditable rationale explaining why a property was valued at a particular price.
- **Absence of Contextual Intelligence:** A single price number is insufficient. Decision-makers need to know where the asset sits within the city distribution, whether price-per-square-foot premiums are justified, and what physical modifications would maximize returns.
- **Data Leakage in ML Pipelines:** Many academic implementations compute global statistics (e.g., standard scaling, target encodings) across the entire dataset before splitting, leading to overly optimistic and unreplicable test results.

---

## 3. Proposed Solution

This system resolves these challenges through an integrated production pipeline:
1. **Leak-Free ML Pipeline:** Scikit-learn Pipeline encapsulating domain feature extraction, median imputation, robust/standard scaling, and one-hot encoding fitted strictly on training folds.
2. **Tuned Gradient Boosted Decision Trees:** Hyperparameter-optimized ensemble model capturing non-linear interactions and cross-feature splits.
3. **Transparent Explainability (SHAP TreeExplainer):** Decomposes every individual valuation into exact marginal INR contributions, separating positive drivers from price discounts.
4. **Market Context & Investment Intelligence:** Automatically contextualizes valuations against city medians, computes percentile distributions, and highlights comparable historical sales.
5. **Interactive What-If Simulation:** Enables investors to test structural renovations (e.g., adding a bathroom, furnishing, parking) and observe predicted valuation delta in real time.

---

## 4. Key Features

- **Accurate Property Valuation:** Instantaneous market valuation in both Indian Lakhs (₹) and Crores (₹ Cr) with empirical ±1σ (68%) and ±2σ (95%) confidence intervals.
- **Explainable AI Engine:** Global feature importance ranking alongside localized waterfall SHAP decomposition showing exact rupee additions/deductions relative to the baseline market average.
- **Market Intelligence & Percentile Benchmarking:** Dynamic comparison against metropolitan averages, price-per-square-foot benchmarks, and historical comparable transactions.
- **Investment Viability Scoring:** Algorithmic Valuation Quality Score (0–100) and automated executive summary classifying assets into tiers (*Prime Investment Asset*, *High Quality Residential*, *Balanced Core Market*, or *Opportunity / Value Asset*).
- **Interactive What-If Scenario Modeler:** Real-time simulation of structural or amenity modifications to forecast marginal return on investment.
- **Enterprise-Ready REST API:** Fully documented OpenAPI/Swagger endpoints with strict Pydantic v2 data validation and sub-50ms response latencies.

---

## 5. System Architecture

```
+-----------------------------------------------------------------------------------+
|                            CLIENT / PRESENTATION LAYER                            |
|     Responsive Modern Web Dashboard (Vanilla HTML5 / CSS3 / ES6 JavaScript)       |
|   [Overview]  [Property Valuation]  [Prediction]  [AI Insights]  [Explainable AI]  |
+-----------------------------------------------------------------------------------+
                                         │  HTTP / JSON (REST)
                                         ▼
+-----------------------------------------------------------------------------------+
|                             FASTAPI APPLICATION SERVER                            |
|                       (Uvicorn ASGI Server / Port 8000)                           |
|  - CORS Middleware              - Pydantic Request Validation & Schemas           |
|  - Static File Mounting         - Centralized Exception Handling                  |
+-----------------------------------------------------------------------------------+
        │                                 │                                 │
        ▼                                 ▼                                 ▼
+─────────────────+             +──────────────────+             +──────────────────+
|  MODEL SERVICE  |             |  EXPLAINABILITY  |             | MARKET INTELLIG. |
|  (model_service)|             |  (TreeExplainer) |             | (dataset stats)  |
+─────────────────+             +──────────────────+             +──────────────────+
        │                                 │                                 │
        └────────────────────────┬────────┴─────────────────────────────────┘
                                 ▼
+-----------------------------------------------------------------------------------+
|                        MACHINE LEARNING PIPELINE (.joblib)                        |
|  - Custom Feature Engineering (Floor Ratio, Area Per BHK, Amenity Score, etc.)    |
|  - ColumnTransformer (One-Hot Encoding, Median Imputation, Robust/Standard Scale) |
|  - Tuned GradientBoostingRegressor (100 estimators, max_depth=6, lr=0.1)          |
+-----------------------------------------------------------------------------------+
                                 │
                                 ▼
+-----------------------------------------------------------------------------------+
|                          PERSISTED ARTIFACTS & DATA STORAGE                       |
|   models/real_estate_gbr_pipeline.joblib   │   data/sample_train.csv / train.csv  |
+-----------------------------------------------------------------------------------+
```

---

## 6. Machine Learning Methodology

The machine learning workflow follows industry-standard CRISP-DM methodology:
1. **Exploratory Data Analysis (EDA):** Univariate distribution profiling, spatial price clustering, multi-collinearity checks, and residual variance analysis.
2. **Leak-Free Partitioning:** 80/20 train/test split (64,000 train, 16,000 test) using a fixed random seed (`random_state=42`).
3. **Domain Feature Construction:** Engineering spatial ratios, transit density metrics, and structural efficiency indices prior to transformation.
4. **Pipeline Encapsulation:** All transformations are bound inside a Scikit-Learn `Pipeline` ensuring zero test-set contamination during inference.
5. **K-Fold Cross-Validation:** 5-fold cross-validation during hyperparameter exploration.
6. **Empirical Calibration:** Calculation of residual standard deviation ($\sigma_{\text{residuals}} \approx 70.23$ Lakhs) to construct realistic confidence bounds.

---

## 7. Dataset Description & Features

The model was trained on an extensive dataset of **80,000 verified residential properties** across **10 tier-1 Indian metropolitan cities**: *Mumbai, Delhi, Bangalore, Pune, Hyderabad, Chennai, Kolkata, Ahmedabad, Chandigarh,* and *Jaipur*.

### Feature Breakdown

| Feature | Type | Description | Range / Values |
| :--- | :--- | :--- | :--- |
| `City` | Categorical | Metropolitan city market | 10 Indian Metros |
| `Locality_Type` | Categorical | Neighborhood demographic tier | `Affordable`, `Mid-Range`, `Premium` |
| `Property_Type` | Categorical | Architectural housing format | `Apartment`, `Builder Floor`, `Independent House`, `Villa` |
| `BHK` | Integer | Bedroom count | 1 to 5 |
| `Bathrooms` | Integer | Bathroom count | 1 to 6 |
| `Super_Area_SqFt` | Continuous | Total built-up architectural area | 500 to 5,000 SqFt |
| `Carpet_Area_SqFt` | Continuous | Usable internal living space | 300 to 4,500 SqFt |
| `Floor_Number` | Integer | Vertical elevation level | 0 to 50 |
| `Total_Floors` | Integer | Total building height | 1 to 50 |
| `Age_of_Property` | Integer | Age of building since construction | 0 to 50 Years |
| `Furnishing_Status`| Categorical | Interior fit-out state | `Unfurnished`, `Semi-Furnished`, `Furnished` |
| `Parking` | Binary | Dedicated vehicle parking slot | 0 or 1 |
| `Lift_Available` | Binary | Operational passenger elevator | 0 or 1 |
| `Gated_Community` | Binary | 24/7 guarded security perimeter | 0 or 1 |
| `Distance_to_Metro_km` | Continuous | Proximity to nearest metro station | 0.1 to 30.0 km |
| `Distance_to_City_Center_km` | Continuous | Distance to primary CBD | 0.5 to 45.0 km |
| **`Price_INR_Lakhs`** | **Continuous** | **Target: Property market price** | **₹ 10.0 to ₹ 950.0 Lakhs** |

---

## 8. Data Preprocessing & Leakage Prevention

To guarantee complete model integrity and prevent data snooping:
- **Zero Information Leakage:** Imputers and scalers are strictly fitted on the training split ($N=64,000$) and only applied (transformed) on the evaluation split ($N=16,000$).
- **One-Hot Encoding with Strict Schemas:** Categorical variables (`City`, `Locality_Type`, `Property_Type`, `Furnishing_Status`) are encoded using `OneHotEncoder(handle_unknown='ignore')` to gracefully handle unseen variants during production serving.
- **Missing Value Imputation:** Handled natively via `SimpleImputer(strategy='median')` for continuous numerical inputs.
- **Robust Feature Scaling:** Applied `StandardScaler` inside the pipeline step to prevent high-magnitude area metrics from dominating gradient step updates.

---

## 9. Feature Engineering Strategy

Domain-specific interactions were engineered to capture spatial and architectural dynamics:

1. **Carpet-to-Super Area Ratio (`Carpet_Super_Ratio`):**
   $$\text{Ratio} = \frac{\text{Carpet\_Area\_SqFt}}{\text{Super\_Area\_SqFt}}$$
   Measures floorplan efficiency; penalizes excessive common area loading.
2. **Area Differential (`Area_Diff_SqFt`):**
   $$\Delta_{\text{Area}} = \text{Super\_Area\_SqFt} - \text{Carpet\_Area\_SqFt}$$
   Represents unlivable shared infrastructure area.
3. **Space Allocation per Bedroom (`Area_per_BHK`):**
   $$\text{Area\_per\_BHK} = \frac{\text{Super\_Area\_SqFt}}{\text{BHK}}$$
   Differentiates cramped budget apartments from spacious luxury layouts.
4. **Space Allocation per Bathroom (`Area_per_Bathroom`):**
   $$\text{Area\_per\_Bathroom} = \frac{\text{Super\_Area\_SqFt}}{\text{Bathrooms}}$$
5. **Vertical Elevation Ratio (`Floor_Ratio`):**
   $$\text{Floor\_Ratio} = \frac{\text{Floor\_Number}}{\text{Total\_Floors}}$$
   Measures elevation positioning within the structure (penthouse vs ground floor).
6. **Composite Amenity Score (`Amenity_Score`):**
   $$\text{Score} = \text{Parking} + \text{Lift\_Available} + \text{Gated\_Community}$$
   Aggregates basic building conveniences into an integer scale (0–3).
7. **Composite Transit Distance (`Total_Distance_km`):**
   $$\text{Distance} = \text{Distance\_to\_Metro\_km} + \text{Distance\_to\_City\_Center\_km}$$
8. **Transit-to-Center Ratio (`Distance_Ratio`):**
   $$\text{Distance\_Ratio} = \frac{\text{Distance\_to\_Metro\_km}}{\text{Distance\_to\_City\_Center\_km} + 0.1}$$

---

## 10. Model Selection & Training

Multiple candidate regression algorithms were evaluated during preliminary benchmarking:
1. **Ordinary Least Squares (OLS) / Ridge / Lasso:** Failed to model sharp regional step-functions and non-linear interactions between city location and square footage.
2. **Random Forest Regressor:** Strong baseline but suffered from deep tree bloat and limited extrapolation at high-valuation tails.
3. **Gradient Boosting Regressor (GBR):** Demonstrated superior empirical performance by sequentially minimizing residual loss via gradient descent in function space, achieving optimal balance between accuracy, model size, and inference latency.

---

## 11. Hyperparameter Tuning & Validation

The baseline Gradient Boosting model was tuned using **5-Fold Cross-Validation** over a targeted grid focusing on tree depth, learning rate, and sub-sampling regularization:

### Optimized Parameter Configuration
- `n_estimators`: `100`
- `learning_rate`: `0.1`
- `max_depth`: `6` (increased from default 3 to capture higher-order regional and area interactions)
- `min_samples_split`: `5`
- `min_samples_leaf`: `4`
- `subsample`: `0.9` (stochastic gradient subsampling to prevent overfitting)
- `loss`: `'squared_error'`

---

## 12. Model Performance & Comparison Table

Evaluation conducted on the **strictly untouched holdout test set (16,000 properties)**:

| Metric | Baseline Gradient Boosting | Tuned Gradient Boosting | Absolute Improvement | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | ₹ 38.4107 Lakhs | **₹ 33.3051 Lakhs** | **-5.1056 Lakhs** | **13.29% Error Reduction** |
| **Mean Squared Error (MSE)** | 6004.5140 | **4932.4408** | **-1072.0732** | **17.85% Variance Reduction** |
| **Root Mean Squared Error (RMSE)** | ₹ 77.4888 Lakhs | **₹ 70.2313 Lakhs** | **-7.2575 Lakhs** | **9.37% Error Reduction** |
| **Coefficient of Determination ($R^2$)**| 0.7288 | **0.7772** | **+0.0484** | **+6.64% Fit Improvement** |

### Cross-Validation Stability
- **5-Fold Mean $R^2$:** `0.7761` ($\pm 0.0034$)
- Consistent metrics across folds confirm absence of overfitting and high generalization capability across unseen property profiles.

---

## 13. Explainable AI & SHAP Interpretation

To ensure full transparency in automated valuation, the system integrates **SHAP (SHapley Additive exPlanations)** based on cooperative game theory:

1. **Global Importance:**
   - **`Super_Area_SqFt` & `Carpet_Area_SqFt`** account for over 52% of total feature attribution.
   - **`City_Mumbai`** exerts the largest single regional premium coefficient (positive SHAP push up to +₹120 Lakhs).
   - **`Locality_Type_Premium`** provides consistent positive marginal valuation uplift.
   - **`Distance_to_City_Center_km`** acts as the primary global depreciation driver.
2. **Local Waterfall Decomposition:**
   - Every inference response decomposes the property's final prediction into:
     $$\text{Predicted Price} = \text{Base Value } (\approx ₹ 142.51 \text{ L}) + \sum_{i=1}^{M} \phi_i$$
     where $\phi_i$ is the exact positive or negative rupee contribution of feature $i$.
   - Positive drivers and discount factors are highlighted directly in the user interface.

---

## 14. Real Estate Domain Intelligence Engine

Beyond raw price prediction, the domain intelligence engine computes:
- **Valuation Quality Index (0–100):** A composite score factoring space efficiency, locality tier, floor elevation advantage, building age depreciation, and amenity density.
- **Asset Tier Classification:**
  - `Prime Investment Asset` (Score $\ge 85$)
  - `High Quality Residential` (Score $70 - 84$)
  - `Balanced Core Market` (Score $50 - 69$)
  - `Opportunity / Value Asset` (Score $< 50$)
- **Price-Per-Square-Foot Valuation:** Computes unit rates for both Super Area and Carpet Area, comparing them with localized city benchmarks.
- **Comparable Properties Search:** Identifies similar historical sales within the same metropolitan territory and property category.

---

## 15. Backend Architecture & FastAPI Endpoints

The backend is built with **FastAPI** and served via **Uvicorn**, providing fully validated REST endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health status and uptime indicator |
| `GET` | `/model-info` | Metadata, hyperparameters, and verified test metrics |
| `GET` | `/features` | Feature definitions, categorical options, and default values |
| `GET` | `/overview-metrics` | Macro portfolio statistics, total listings, and city benchmarks |
| `POST` | `/predict` | Full valuation inference with confidence ranges & investment score |
| `POST` | `/explain` | Global and local SHAP feature attribution decomposition |
| `POST` | `/market-comparison` | Property percentile standing and city median comparison |
| `POST` | `/comparable-properties`| Top historical comparable sales for the submitted asset |
| `POST` | `/what-if` | Real-time valuation delta for structural/amenity modifications |

### Example Request (`POST /predict`)
```json
{
  "City": "Mumbai",
  "Locality_Type": "Premium",
  "Property_Type": "Apartment",
  "BHK": 3,
  "Bathrooms": 3,
  "Super_Area_SqFt": 1500.0,
  "Carpet_Area_SqFt": 1200.0,
  "Floor_Number": 12,
  "Total_Floors": 28,
  "Age_of_Property": 3,
  "Furnishing_Status": "Furnished",
  "Parking": 1,
  "Lift_Available": 1,
  "Gated_Community": 1,
  "Distance_to_Metro_km": 1.2,
  "Distance_to_City_Center_km": 4.5
}
```

---

## 16. Frontend User Interface Structure

The frontend is a lightweight, responsive Single Page Application (SPA) built with semantic HTML5, modern CSS custom properties (variables), and modular ES6 JavaScript.

The navigation consists strictly of 6 dedicated core pages:
1. **🏠 Overview:** Macro market dashboard, total listings analyzed, median pricing, and model accuracy highlights.
2. **💰 Property Valuation:** Interactive valuation input console with real-time field validation and sample presets.
3. **🎯 Prediction Result:** Primary valuation cards displaying price in Lakhs & Crores, confidence bounds, and investment tier.
4. **🧠 AI Insights:** Strategic investment analysis, asset quality score, key value drivers, and discount factors.
5. **🔍 Explainable AI:** Interactive SHAP local impact waterfall and top 15 global feature importance bars.
6. **📊 Market Intelligence:** City percentile distribution, unit rate comparisons, historical comparables, and interactive What-If scenario simulator.

---

## 17. Visualizations & Diagnostic Plots

Comprehensive diagnostic plots are saved in `visualizations/`:

| Visualization | File Name | Description |
| :--- | :--- | :--- |
| **Actual vs Predicted** | `actual_vs_predicted.png` | Scatter plot with $y=x$ ideal reference line demonstrating test accuracy |
| **Residual Distribution** | `residual_distribution.png` | Histogram & KDE showing zero-centered, normally distributed residuals |
| **Residual vs Predicted** | `residual_plot.png` | Homoscedasticity evaluation across price ranges |
| **SHAP Summary Plot** | `shap_summary_plot.png` | Beeswarm plot of feature impacts and directions |
| **Feature Importance** | `feature_importance_top15.png` | Top 15 GBR split-importance ranking |
| **City Median Prices** | `city_median_prices.png` | Median price comparison across all 10 metros |
| **Price vs Area** | `price_vs_area.png` | Area vs Price regression line and dispersion |
| **Correlation Heatmap** | `correlation_heatmap.png` | Feature-to-feature Pearson correlation matrix |
| **Error by City** | `error_by_city.png` | Residual variance categorized by urban center |
| **Error by Property Type**| `error_by_property_type.png` | Residual error across Apartment, Villa, etc. |

---

## 18. Technology Stack

- **Machine Learning & Analytics:** Python 3.10+, Scikit-Learn 1.3+, Pandas 2.0+, NumPy 1.24+, Joblib 1.3+
- **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations) 0.42+
- **Backend API:** FastAPI 0.100+, Uvicorn (ASGI) 0.22+, Pydantic v2
- **Data Visualization:** Matplotlib 3.7+, Seaborn 0.12+
- **Frontend UI:** HTML5, CSS3, Vanilla ES6 JavaScript (zero third-party JS framework overhead)

---

## 19. Project Directory Structure

```
├── backend/
│   ├── main.py                  # FastAPI server, endpoints, and route handlers
│   ├── model_service.py         # Model loading, prediction, and market intelligence logic
│   ├── explainability.py        # SHAP TreeExplainer and feature attribution engine
│   ├── schemas.py               # Pydantic v2 request/response validation schemas
│   └── requirements.txt         # Backend service dependencies
├── data/
│   └── sample_train.csv         # Stratified 500-record sample dataset for immediate execution
├── frontend/
│   ├── index.html               # 6-page responsive user dashboard interface
│   ├── style.css                # Premium modern SaaS design system
│   └── script.js                # Asynchronous API client, DOM rendering, What-If simulator
├── models/
│   └── real_estate_gbr_pipeline.joblib  # Trained, tuned ML pipeline artifact (~163 KB)
├── notebooks/
│   └── eda_and_modeling.ipynb   # Comprehensive Jupyter notebook for EDA and model training
├── src/
│   ├── evaluate.py              # Metrics evaluation, residual analysis, CV scoring
│   ├── feature_engineering.py   # Leak-free feature transformer classes
│   ├── intelligence.py          # Domain investment rules and asset scoring logic
│   ├── predict.py               # Standalone prediction wrapper
│   ├── preprocessing.py         # ColumnTransformer and categorical encoders
│   └── train.py                 # Full training, cross-validation, and tuning pipeline
├── visualizations/              # 14 high-resolution diagnostic and EDA PNG plots
├── .gitignore                   # Excludes __pycache__, envs, large train.csv, secrets
├── create_notebook.py           # Programmatic notebook generation script
├── generate_eda.py              # Script to reproduce all 14 diagnostic visualizations
├── requirements.txt             # Root project dependencies
├── test_api.py                  # Automated backend and endpoint verification suite
└── README.md                    # Comprehensive project documentation
```

---

## 20. Installation & Setup Guide

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd "REA ESTAE.CSV"
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Dataset Setup (Optional for Full Retraining)
- The repository includes `data/sample_train.csv` so the entire application functions **out of the box** immediately upon cloning.
- If you wish to retrain the model on the full 80,000-listing dataset, place `train.csv` in the project root directory and run `python src/train.py`.

---

## 21. Usage Guide & API Testing

### 1. Start the Application Server
Run the FastAPI backend with hot-reload enabled:
```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 2. Access the Application
- **Interactive Web Interface:** Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- **Interactive API Documentation (Swagger UI):** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
- **Alternative API Docs (ReDoc):** Open [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).

### 3. Run Automated Endpoint Verification
Verify all API endpoints using the built-in test suite:
```bash
python test_api.py
```

### 4. Example cURL Prediction Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "City": "Bangalore",
       "Locality_Type": "Premium",
       "Property_Type": "Apartment",
       "BHK": 3,
       "Bathrooms": 3,
       "Super_Area_SqFt": 1750.0,
       "Carpet_Area_SqFt": 1400.0,
       "Floor_Number": 14,
       "Total_Floors": 24,
       "Age_of_Property": 2,
       "Furnishing_Status": "Furnished",
       "Parking": 1,
       "Lift_Available": 1,
       "Gated_Community": 1,
       "Distance_to_Metro_km": 0.8,
       "Distance_to_City_Center_km": 6.2
     }'
```

---

## 22. Key Insights & Business Takeaways

1. **Space is King, but Location Dictates the Multiple:** While area features dominate raw importance, the `City` feature acts as an exponential multiplier. A 1,500 SqFt apartment in Mumbai commands over $2.8\times$ the valuation of an identical unit in Kolkata or Ahmedabad.
2. **Transit Access Yields Diminishing Returns:** Proximity to a metro station provides significant value uplift when within 2 km, but benefits plateau beyond that threshold.
3. **Floor Elevation Advantage:** In high-density metros, upper floors (high `Floor_Ratio`) command a consistent 4–8% valuation premium due to reduced noise and superior ventilation.
4. **Actionable Commercial Impact:** By replacing arbitrary heuristics with transparent SHAP explanations and empirical confidence bounds, the system provides mortgage lenders and institutional property funds with defensible, auditable valuation reports.

---

## License
This project is open-source and licensed under the [MIT License](LICENSE).
