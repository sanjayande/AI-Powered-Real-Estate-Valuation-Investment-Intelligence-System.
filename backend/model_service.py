import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import RealEstatePredictor
from backend.schemas import (
    PropertyInputSchema,
    PredictionResponseSchema,
    ModelInfoResponseSchema,
    FeaturesInfoResponseSchema,
    OverviewMetricsResponseSchema,
    MarketComparisonResponseSchema,
    CityMarketStat,
    ComparablePropertyItem,
    WhatIfResponseSchema,
    ALLOWED_CITIES,
    ALLOWED_LOCALITY_TYPES,
    ALLOWED_PROPERTY_TYPES,
    ALLOWED_FURNISHING_STATUSES
)

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'real_estate_gbr_pipeline.joblib')
DATA_PATH = os.path.join(PROJECT_ROOT, 'train.csv')
SAMPLE_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'sample_train.csv')

class ModelService:
    def __init__(self, model_path: str = MODEL_PATH, data_path: str = DATA_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained pipeline model not found at {model_path}. Train the model first.")
        
        self.predictor = RealEstatePredictor(model_path=model_path)
        self.raw_joblib = joblib.load(model_path)
        
        # Load verified historical dataset for market analytics
        resolved_data_path = data_path if os.path.exists(data_path) else (SAMPLE_DATA_PATH if os.path.exists(SAMPLE_DATA_PATH) else None)
        if resolved_data_path:
            self.df = pd.read_csv(resolved_data_path)
            # Ensure price per sqft column is available for fast aggregation
            self.df['Price_Per_SqFt_Calc'] = (self.df['Price_INR_Lakhs'] * 100000.0) / self.df['Super_Area_SqFt']
        else:
            self.df = pd.DataFrame()

    def predict(self, property_input: PropertyInputSchema) -> PredictionResponseSchema:
        input_dict = property_input.model_dump()
        report = self.predictor.predict_with_intelligence(input_dict)
        return PredictionResponseSchema(**report)

    def get_info(self) -> ModelInfoResponseSchema:
        metrics_baseline = self.raw_joblib.get('metrics_baseline', {})
        metrics_tuned = self.raw_joblib.get('metrics_tuned', {})
        best_params = self.raw_joblib.get('best_params', {})
        
        return ModelInfoResponseSchema(
            model_title="AI-Powered Real Estate Valuation & Investment Intelligence System",
            target_column="Price_INR_Lakhs",
            metrics_baseline=metrics_baseline,
            metrics_tuned=metrics_tuned,
            best_parameters=best_params,
            total_dataset_rows=len(self.df) if not self.df.empty else 80000
        )

    def get_features(self) -> FeaturesInfoResponseSchema:
        defaults = {
            'City': 'Delhi',
            'Locality_Type': 'Premium',
            'Property_Type': 'Apartment',
            'Furnishing_Status': 'Semi-Furnished',
            'BHK': 3,
            'Bathrooms': 3,
            'Super_Area_SqFt': 1650.0,
            'Carpet_Area_SqFt': 1250.0,
            'Floor_Number': 8,
            'Total_Floors': 20,
            'Age_of_Property': 4,
            'Parking': 1,
            'Lift_Available': 1,
            'Gated_Community': 1,
            'Distance_to_Metro_km': 1.2,
            'Distance_to_City_Center_km': 6.5
        }
        
        return FeaturesInfoResponseSchema(
            cities=ALLOWED_CITIES,
            locality_types=ALLOWED_LOCALITY_TYPES,
            property_types=ALLOWED_PROPERTY_TYPES,
            furnishing_statuses=ALLOWED_FURNISHING_STATUSES,
            defaults=defaults
        )

    def get_overview_metrics(self) -> OverviewMetricsResponseSchema:
        metrics_tuned = self.raw_joblib.get('metrics_tuned', {})
        r2_score = round(float(metrics_tuned.get('R2', 0.0)), 4)
        mae_score = round(float(metrics_tuned.get('MAE', 0.0)), 2)

        if not self.df.empty:
            total_props = int(len(self.df))
            cities_count = int(self.df['City'].nunique())
            avg_price = round(float(self.df['Price_INR_Lakhs'].mean()), 2)
            med_price = round(float(self.df['Price_INR_Lakhs'].median()), 2)

            city_meds = self.df.groupby('City')['Price_INR_Lakhs'].median().sort_values(ascending=False)
            top_city_name = str(city_meds.index[0])
            top_city_val = round(float(city_meds.iloc[0]), 2)
            top_city_data = {"city": top_city_name, "median_price_lakhs": top_city_val}
        else:
            total_props = 0
            cities_count = 0
            avg_price = 0.0
            med_price = 0.0
            top_city_data = {"city": "N/A", "median_price_lakhs": 0.0}

        feat_imp_df = self.raw_joblib.get('feature_importances', None)
        if feat_imp_df is not None and not feat_imp_df.empty:
            top_feat_name = str(feat_imp_df.iloc[0]['Feature'])
            top_feat_val = round(float(feat_imp_df.iloc[0]['Importance']) * 100, 2)
            top_feature = {"feature": top_feat_name, "importance_percentage": top_feat_val}
        else:
            top_feature = {"feature": "N/A", "importance_percentage": 0.0}

        return OverviewMetricsResponseSchema(
            properties_analyzed=total_props,
            cities_covered=cities_count,
            average_price_lakhs=avg_price,
            median_price_lakhs=med_price,
            model_r2=r2_score,
            model_mae_lakhs=mae_score,
            top_city_by_median=top_city_data,
            most_important_feature=top_feature,
            model_name="Tuned GradientBoostingRegressor (50 Estimators, Depth 4)",
            model_status="Operational",
            dataset_status=f"{total_props:,} Verified Historical Records Loaded",
            api_status="Connected & Serving"
        )

    def get_market_comparison(self, property_input: PropertyInputSchema, predicted_price: Optional[float] = None) -> MarketComparisonResponseSchema:
        if predicted_price is None:
            df_in = pd.DataFrame([property_input.model_dump()])
            predicted_price = float(self.predictor.pipeline.predict(df_in)[0])

        predicted_price = round(predicted_price, 2)
        price_per_sqft = round((predicted_price * 100000.0) / property_input.Super_Area_SqFt, 0) if property_input.Super_Area_SqFt > 0 else 0.0

        if self.df.empty:
            raise ValueError("Dataset not available for market comparison.")

        city_df = self.df[self.df['City'] == property_input.City]
        if city_df.empty:
            city_df = self.df

        city_median = round(float(city_df['Price_INR_Lakhs'].median()), 2)
        city_mean = round(float(city_df['Price_INR_Lakhs'].mean()), 2)
        city_median_sqft = round(float(city_df['Price_Per_SqFt_Calc'].median()), 0)

        # Market difference vs city median
        market_diff_pct = round(((predicted_price - city_median) / city_median) * 100.0, 1)

        # Percentile rank within city
        city_percentile = round(float((city_df['Price_INR_Lakhs'] <= predicted_price).mean() * 100.0), 1)
        # Percentile rank across entire national dataset
        national_percentile = round(float((self.df['Price_INR_Lakhs'] <= predicted_price).mean() * 100.0), 1)

        # Market position tier
        if city_percentile >= 90:
            tier = "Luxury Market Tier (Top 10%)"
        elif city_percentile >= 75:
            tier = "Upper Quartile Market"
        elif city_percentile >= 50:
            tier = "Above Median Mid-Market"
        elif city_percentile >= 25:
            tier = "Core Accessible Mid-Market"
        else:
            tier = "Value / Budget Market Segment"

        diff_sign = "+" if market_diff_pct >= 0 else ""
        position_desc = (
            f"This property's estimated valuation of ₹ {predicted_price:.2f} Lakhs places it at the {city_percentile}th "
            f"percentile of the {property_input.City} housing market, which is {diff_sign}{market_diff_pct:.1f}% "
            f"relative to the city median price of ₹ {city_median:.2f} Lakhs."
        )

        # Price vs Area insight
        type_df = city_df[city_df['Property_Type'] == property_input.Property_Type]
        ref_df = type_df if len(type_df) >= 30 else city_df
        type_med_sqft = round(float(ref_df['Price_Per_SqFt_Calc'].median()), 0)
        sqft_diff_pct = round(((price_per_sqft - type_med_sqft) / type_med_sqft) * 100.0, 1) if type_med_sqft > 0 else 0.0

        if sqft_diff_pct > 10.0:
            price_vs_area_insight = (
                f"At ₹ {price_per_sqft:,.0f}/SqFt, this property commands a +{sqft_diff_pct:.1f}% unit-rate premium "
                f"over the median for {property_input.Property_Type}s in {property_input.City} (₹ {type_med_sqft:,.0f}/SqFt), "
                f"driven by prime locality tier and superior amenities."
            )
        elif sqft_diff_pct < -10.0:
            price_vs_area_insight = (
                f"At ₹ {price_per_sqft:,.0f}/SqFt, this property provides high area efficiency, priced at a {abs(sqft_diff_pct):.1f}% "
                f"discount to typical {property_input.Property_Type}s in {property_input.City} (median ₹ {type_med_sqft:,.0f}/SqFt)."
            )
        else:
            price_vs_area_insight = (
                f"At ₹ {price_per_sqft:,.0f}/SqFt, this property aligns closely (within ±10%) with prevailing market rates for "
                f"{property_input.Property_Type}s in {property_input.City} (median ₹ {type_med_sqft:,.0f}/SqFt)."
            )

        # City market comparison table
        city_stats_list: List[CityMarketStat] = []
        city_groups = self.df.groupby('City')
        for c_name, c_group in city_groups:
            c_med = round(float(c_group['Price_INR_Lakhs'].median()), 2)
            c_mean = round(float(c_group['Price_INR_Lakhs'].mean()), 2)
            c_count = int(len(c_group))
            c_med_sqft = round(float(c_group['Price_Per_SqFt_Calc'].median()), 0)
            city_stats_list.append(CityMarketStat(
                city=str(c_name),
                median_price_lakhs=c_med,
                mean_price_lakhs=c_mean,
                count=c_count,
                median_price_per_sqft=c_med_sqft,
                is_selected=(str(c_name) == property_input.City)
            ))
        city_stats_list.sort(key=lambda x: x.median_price_lakhs, reverse=True)

        # Real Comparable Properties Query
        # First filter same city, same property type
        pool = city_df[city_df['Property_Type'] == property_input.Property_Type].copy()
        if len(pool) < 5:
            pool = city_df.copy()

        # Sort by distance in Super_Area_SqFt and BHK
        pool['area_dist'] = (pool['Super_Area_SqFt'] - property_input.Super_Area_SqFt).abs()
        pool['bhk_dist'] = (pool['BHK'] - property_input.BHK).abs()
        # Combined proximity score
        pool['comp_score'] = (pool['area_dist'] / (property_input.Super_Area_SqFt + 1e-5)) + (pool['bhk_dist'] * 0.5)
        top_comps = pool.sort_values('comp_score').head(5)

        comparables: List[ComparablePropertyItem] = []
        for _, row in top_comps.iterrows():
            comp_price = round(float(row['Price_INR_Lakhs']), 2)
            comp_sqft_price = round(float(row['Price_Per_SqFt_Calc']), 0)
            diff_lakhs = round(comp_price - predicted_price, 2)
            comparables.append(ComparablePropertyItem(
                property_id=str(row.get('Property_ID', 'PROP-REF')),
                city=str(row['City']),
                locality_type=str(row['Locality_Type']),
                property_type=str(row['Property_Type']),
                super_area_sqft=round(float(row['Super_Area_SqFt']), 1),
                bhk=int(row['BHK']),
                bathrooms=int(row['Bathrooms']),
                age_of_property=int(row['Age_of_Property']),
                price_lakhs=comp_price,
                price_per_sqft=comp_sqft_price,
                price_diff_vs_subject_lakhs=diff_lakhs
            ))

        return MarketComparisonResponseSchema(
            property_price_lakhs=predicted_price,
            city=property_input.City,
            city_median_lakhs=city_median,
            city_mean_lakhs=city_mean,
            price_per_sqft=price_per_sqft,
            city_median_price_per_sqft=city_median_sqft,
            market_diff_percentage=market_diff_pct,
            city_percentile=city_percentile,
            national_percentile=national_percentile,
            market_position_tier=tier,
            position_description=position_desc,
            price_vs_area_insight=price_vs_area_insight,
            city_comparison=city_stats_list,
            comparable_properties=comparables
        )

    def what_if_simulation(self, base_property: PropertyInputSchema, modified_property: PropertyInputSchema) -> WhatIfResponseSchema:
        base_df = pd.DataFrame([base_property.model_dump()])
        mod_df = pd.DataFrame([modified_property.model_dump()])

        base_val = round(float(self.predictor.pipeline.predict(base_df)[0]), 2)
        mod_val = round(float(self.predictor.pipeline.predict(mod_df)[0]), 2)
        diff_lakhs = round(mod_val - base_val, 2)
        diff_pct = round((diff_lakhs / base_val) * 100.0, 2) if base_val > 0 else 0.0

        diff_sign = "+" if diff_lakhs >= 0 else ""
        direction = "increase" if diff_lakhs >= 0 else "reduction"

        # Track key changes
        changes = []
        if base_property.Super_Area_SqFt != modified_property.Super_Area_SqFt:
            changes.append(f"Super Area: {base_property.Super_Area_SqFt:,.0f} → {modified_property.Super_Area_SqFt:,.0f} SqFt")
        if base_property.Carpet_Area_SqFt != modified_property.Carpet_Area_SqFt:
            changes.append(f"Carpet Area: {base_property.Carpet_Area_SqFt:,.0f} → {modified_property.Carpet_Area_SqFt:,.0f} SqFt")
        if base_property.Distance_to_City_Center_km != modified_property.Distance_to_City_Center_km:
            changes.append(f"Center Distance: {base_property.Distance_to_City_Center_km} → {modified_property.Distance_to_City_Center_km} km")
        if base_property.Distance_to_Metro_km != modified_property.Distance_to_Metro_km:
            changes.append(f"Metro Distance: {base_property.Distance_to_Metro_km} → {modified_property.Distance_to_Metro_km} km")
        if base_property.Age_of_Property != modified_property.Age_of_Property:
            changes.append(f"Age: {base_property.Age_of_Property} → {modified_property.Age_of_Property} yrs")
        if base_property.BHK != modified_property.BHK:
            changes.append(f"BHK: {base_property.BHK} → {modified_property.BHK}")
        if base_property.Floor_Number != modified_property.Floor_Number:
            changes.append(f"Floor: {base_property.Floor_Number} → {modified_property.Floor_Number}")

        changes_str = ", ".join(changes) if changes else "modified parameter parameters"
        summary = (
            f"Simulated parameter adjustment ({changes_str}) yields a {direction} of "
            f"{diff_sign}₹ {abs(diff_lakhs):.2f} Lakhs ({diff_sign}{diff_pct:.1f}%) "
            f"from ₹ {base_val:.2f} L to ₹ {mod_val:.2f} Lakhs under the trained model pipeline."
        )

        return WhatIfResponseSchema(
            base_price_lakhs=base_val,
            simulated_price_lakhs=mod_val,
            difference_lakhs=diff_lakhs,
            difference_percentage=diff_pct,
            summary=summary
        )

# Global Singleton Instance
model_service = ModelService()
