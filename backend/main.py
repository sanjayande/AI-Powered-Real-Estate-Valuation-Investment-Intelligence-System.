import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.schemas import (
    PropertyInputSchema,
    PredictionResponseSchema,
    ModelInfoResponseSchema,
    FeaturesInfoResponseSchema,
    ExplainabilityResponseSchema,
    OverviewMetricsResponseSchema,
    MarketComparisonResponseSchema,
    WhatIfRequestSchema,
    WhatIfResponseSchema
)
from backend.model_service import model_service
from backend.explainability import explainability_service

app = FastAPI(
    title="REAL ESTATE AI — Valuation & Intelligence API",
    description="Enterprise PropTech API serving tuned GradientBoostingRegressor valuations, local SHAP explainability, real market intelligence, and what-if simulation.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local development and browser frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files for Visualizations
viz_path = os.path.join(PROJECT_ROOT, 'visualizations')
if os.path.exists(viz_path):
    app.mount("/visualizations", StaticFiles(directory=viz_path), name="visualizations")

# Mount Static Files for Frontend (CSS, JS)
frontend_path = os.path.join(PROJECT_ROOT, 'frontend')
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "REAL ESTATE AI Valuation API is running. Visit /docs for Swagger UI."}

@app.get("/health", summary="Health Check Endpoint")
async def health_check():
    return {"status": "healthy"}

@app.get("/model-info", response_model=ModelInfoResponseSchema, summary="Get Model Metadata & Performance Metrics")
async def get_model_info():
    try:
        return model_service.get_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model info: {str(e)}")

@app.get("/features", response_model=FeaturesInfoResponseSchema, summary="Get Required Property Features & Options")
async def get_features():
    try:
        return model_service.get_features()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch feature options: {str(e)}")

@app.get("/overview-metrics", response_model=OverviewMetricsResponseSchema, summary="Get Executive Dashboard Overview Metrics")
async def get_overview_metrics():
    try:
        return model_service.get_overview_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch overview metrics: {str(e)}")

@app.post("/predict", response_model=PredictionResponseSchema, summary="Predict Property Market Valuation & Valuation Score")
async def predict_property(property_input: PropertyInputSchema):
    try:
        # Business logic validation: Carpet area vs Super area
        if property_input.Carpet_Area_SqFt > property_input.Super_Area_SqFt:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Carpet Area cannot exceed Super Built-up Area."
            )
        
        # Business logic validation: Floor number vs Total floors
        if property_input.Floor_Number > property_input.Total_Floors:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Floor Number cannot exceed Total Floors in building."
            )
            
        prediction_result = model_service.predict(property_input)
        return prediction_result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction engine error: {str(e)}")

@app.post("/explain", response_model=ExplainabilityResponseSchema, summary="Get SHAP & Feature Importance Explainability Metadata")
async def get_explainability(property_input: Optional[PropertyInputSchema] = None):
    try:
        return explainability_service.get_explainability(property_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability engine error: {str(e)}")

@app.post("/market-comparison", response_model=MarketComparisonResponseSchema, summary="Get Real Market Context & Comparable Properties")
async def get_market_comparison(property_input: PropertyInputSchema):
    try:
        return model_service.get_market_comparison(property_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market comparison error: {str(e)}")

@app.post("/what-if", response_model=WhatIfResponseSchema, summary="Simulate Property Valuation Scenario via Trained Pipeline")
async def simulate_what_if(what_if_req: WhatIfRequestSchema):
    try:
        if what_if_req.modified_property.Carpet_Area_SqFt > what_if_req.modified_property.Super_Area_SqFt:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Modified Carpet Area cannot exceed Super Built-up Area."
            )
        if what_if_req.modified_property.Floor_Number > what_if_req.modified_property.Total_Floors:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Modified Floor Number cannot exceed Total Floors in building."
            )
        return model_service.what_if_simulation(what_if_req.base_property, what_if_req.modified_property)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-if simulation error: {str(e)}")
