from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

ALLOWED_CITIES = [
    'Chandigarh', 'Kolkata', 'Delhi', 'Ahmedabad', 'Pune',
    'Hyderabad', 'Mumbai', 'Bangalore', 'Jaipur', 'Chennai'
]
ALLOWED_LOCALITY_TYPES = ['Mid-Range', 'Premium', 'Affordable']
ALLOWED_PROPERTY_TYPES = ['Apartment', 'Villa', 'Builder Floor', 'Independent House']
ALLOWED_FURNISHING_STATUSES = ['Semi-Furnished', 'Unfurnished', 'Furnished']

class PropertyInputSchema(BaseModel):
    City: str = Field(..., example='Delhi', description="City location of the property")
    Locality_Type: str = Field(..., example='Premium', description="Locality classification tier")
    Property_Type: str = Field(..., example='Apartment', description="Property building type")
    BHK: int = Field(..., ge=1, le=10, example=3, description="Number of Bedrooms (BHK)")
    Bathrooms: int = Field(..., ge=1, le=10, example=3, description="Number of Bathrooms")
    Super_Area_SqFt: float = Field(..., gt=50, le=20000, example=1650.0, description="Total Super Built-up Area in SqFt")
    Carpet_Area_SqFt: float = Field(..., gt=30, le=15000, example=1250.0, description="Net usable Carpet Area in SqFt")
    Floor_Number: int = Field(..., ge=0, le=100, example=8, description="Floor level of the property")
    Total_Floors: int = Field(..., ge=1, le=100, example=20, description="Total floors in the building")
    Age_of_Property: int = Field(..., ge=0, le=100, example=4, description="Age of the property in years")
    Furnishing_Status: str = Field(..., example='Semi-Furnished', description="Furnishing level")
    Parking: int = Field(..., ge=0, le=1, example=1, description="Dedicated parking slot (1=Yes, 0=No)")
    Lift_Available: int = Field(..., ge=0, le=1, example=1, description="Elevator access (1=Yes, 0=No)")
    Gated_Community: int = Field(..., ge=0, le=1, example=1, description="Gated community security (1=Yes, 0=No)")
    Distance_to_Metro_km: float = Field(..., ge=0.0, le=50.0, example=1.2, description="Distance to nearest metro in km")
    Distance_to_City_Center_km: float = Field(..., ge=0.0, le=100.0, example=6.5, description="Distance to city center in km")

    @field_validator('City')
    def validate_city(cls, v):
        if v not in ALLOWED_CITIES:
            raise ValueError(f"Invalid City '{v}'. Allowed values: {ALLOWED_CITIES}")
        return v

    @field_validator('Locality_Type')
    def validate_locality(cls, v):
        if v not in ALLOWED_LOCALITY_TYPES:
            raise ValueError(f"Invalid Locality_Type '{v}'. Allowed values: {ALLOWED_LOCALITY_TYPES}")
        return v

    @field_validator('Property_Type')
    def validate_property_type(cls, v):
        if v not in ALLOWED_PROPERTY_TYPES:
            raise ValueError(f"Invalid Property_Type '{v}'. Allowed values: {ALLOWED_PROPERTY_TYPES}")
        return v

    @field_validator('Furnishing_Status')
    def validate_furnishing(cls, v):
        if v not in ALLOWED_FURNISHING_STATUSES:
            raise ValueError(f"Invalid Furnishing_Status '{v}'. Allowed values: {ALLOWED_FURNISHING_STATUSES}")
        return v

class PredictionResponseSchema(BaseModel):
    Predicted_Price_Lakhs: float
    Predicted_Price_Crores: float
    Valuation_Lower_Lakhs: float
    Valuation_Upper_Lakhs: float
    Price_Per_SqFt_Super: float
    Price_Per_SqFt_Carpet: float
    Valuation_Score: float
    Valuation_Tier: str
    Executive_Summary: str
    Key_Value_Drivers: List[str]
    Discount_Factors: List[str]

class ModelInfoResponseSchema(BaseModel):
    model_title: str
    target_column: str
    metrics_baseline: Dict[str, Any]
    metrics_tuned: Dict[str, Any]
    best_parameters: Dict[str, Any]
    total_dataset_rows: int

class FeaturesInfoResponseSchema(BaseModel):
    cities: List[str]
    locality_types: List[str]
    property_types: List[str]
    furnishing_statuses: List[str]
    defaults: Dict[str, Any]

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    description: Optional[str] = None

class LocalContributionItem(BaseModel):
    feature: str
    raw_value: str
    contribution_lakhs: float
    is_positive: bool

class ExplainabilityResponseSchema(BaseModel):
    base_value_lakhs: Optional[float] = None
    predicted_price_lakhs: Optional[float] = None
    net_impact_lakhs: Optional[float] = None
    positive_drivers: Optional[List[LocalContributionItem]] = None
    negative_drivers: Optional[List[LocalContributionItem]] = None
    all_contributions: Optional[List[LocalContributionItem]] = None
    ai_explanation: Optional[str] = None
    feature_importances: List[FeatureImportanceItem]
    shap_summary_plot_url: str
    actual_vs_predicted_plot_url: str
    error_by_city_plot_url: str

class ComparablePropertyItem(BaseModel):
    property_id: str
    city: str
    locality_type: str
    property_type: str
    super_area_sqft: float
    bhk: int
    bathrooms: int
    age_of_property: int
    price_lakhs: float
    price_per_sqft: float
    price_diff_vs_subject_lakhs: float

class CityMarketStat(BaseModel):
    city: str
    median_price_lakhs: float
    mean_price_lakhs: float
    count: int
    median_price_per_sqft: float
    is_selected: bool

class MarketComparisonResponseSchema(BaseModel):
    property_price_lakhs: float
    city: str
    city_median_lakhs: float
    city_mean_lakhs: float
    price_per_sqft: float
    city_median_price_per_sqft: float
    market_diff_percentage: float
    city_percentile: float
    national_percentile: float
    market_position_tier: str
    position_description: str
    price_vs_area_insight: str
    city_comparison: List[CityMarketStat]
    comparable_properties: List[ComparablePropertyItem]

class WhatIfRequestSchema(BaseModel):
    base_property: PropertyInputSchema
    modified_property: PropertyInputSchema

class WhatIfResponseSchema(BaseModel):
    base_price_lakhs: float
    simulated_price_lakhs: float
    difference_lakhs: float
    difference_percentage: float
    summary: str

class OverviewMetricsResponseSchema(BaseModel):
    properties_analyzed: int
    cities_covered: int
    average_price_lakhs: float
    median_price_lakhs: float
    model_r2: float
    model_mae_lakhs: float
    top_city_by_median: Dict[str, Any]
    most_important_feature: Dict[str, Any]
    model_name: str
    model_status: str
    dataset_status: str
    api_status: str
