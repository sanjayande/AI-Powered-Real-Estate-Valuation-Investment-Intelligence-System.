import os
import shap
import pandas as pd
import numpy as np
from typing import Optional, List
from backend.model_service import model_service
from backend.schemas import (
    ExplainabilityResponseSchema,
    FeatureImportanceItem,
    LocalContributionItem,
    PropertyInputSchema
)
from src.train import get_transformed_feature_names

FEATURE_DESCRIPTIONS = {
    'City_Mumbai': 'Mumbai metropolitan capital pricing premium',
    'City_Delhi': 'Delhi NCR capital regional market index',
    'City_Bangalore': 'Bangalore tech corridor valuation premium',
    'City_Pune': 'Pune IT / industrial belt regional pricing',
    'City_Chandigarh': 'Chandigarh planned urban territory index',
    'City_Chennai': 'Chennai metropolitan coastal real estate index',
    'City_Hyderabad': 'Hyderabad cyberabad growth corridor pricing',
    'City_Kolkata': 'Kolkata eastern regional metro market baseline',
    'City_Ahmedabad': 'Ahmedabad commercial center pricing index',
    'City_Jaipur': 'Jaipur heritage urban tier-2 baseline',
    'Super_Area_SqFt': 'Total built-up architectural space footprint',
    'Carpet_Area_SqFt': 'Directly usable internal living floor space',
    'Carpet_Super_Ratio': 'Space efficiency ratio of carpet to super area',
    'Area_Diff_SqFt': 'Common area / loading area differential',
    'Area_per_BHK': 'Spaciousness allocation per bedroom unit',
    'Area_per_Bathroom': 'Spatial allocation per bathroom fixture',
    'Total_Rooms': 'Combined bedroom and bathroom unit count',
    'Locality_Type_Premium': 'Prime high-demand residential neighborhood tier',
    'Locality_Type_Mid-Range': 'Established suburban middle-income neighborhood',
    'Locality_Type_Affordable': 'Value-oriented emerging growth corridor',
    'Distance_to_City_Center_km': 'Proximity radius to central business district',
    'Distance_to_Metro_km': 'Walking/feeder distance to rapid transit station',
    'Total_Distance_km': 'Combined composite transit & center distance',
    'Distance_Ratio': 'Ratio of metro distance to city center distance',
    'Age_of_Property': 'Physical age and structural vintage of building',
    'Floor_Number': 'Vertical elevation and floor level positioning',
    'Total_Floors': 'Total building height and structural density',
    'Floor_Ratio': 'Relative vertical placement in building height',
    'Amenity_Score': 'Composite score of parking, lift, and gated security',
    'Parking': 'Dedicated private motor vehicle parking slot',
    'Lift_Available': 'High-speed elevator access in building',
    'Gated_Community': 'Controlled gated perimeter and security check',
    'Property_Type_Apartment': 'Multi-story residential apartment unit',
    'Property_Type_Villa': 'Independent luxury landed villa development',
    'Property_Type_Builder Floor': 'Low-rise independent builder floor residence',
    'Property_Type_Independent House': 'Standalone private residential bungalow',
    'Furnishing_Status_Furnished': 'Fully furnished interior fit-out',
    'Furnishing_Status_Semi-Furnished': 'Standard semi-furnished cabinetry and fixtures',
    'Furnishing_Status_Unfurnished': 'Bare-shell unfurnished delivery condition'
}

class ExplainabilityService:
    def __init__(self):
        pipeline = model_service.predictor.pipeline
        self.pipeline = pipeline
        self.regressor = pipeline.named_steps['regressor']
        self.preprocessor = pipeline.named_steps['preprocessor']
        self.explainer = shap.TreeExplainer(self.regressor)
        
        # Sample dummy df to resolve transformed feature names
        dummy_df = pd.DataFrame([model_service.get_features().defaults])
        self.transformed_feature_names = get_transformed_feature_names(self.preprocessor, dummy_df)

    def get_explainability(self, property_input: Optional[PropertyInputSchema] = None) -> ExplainabilityResponseSchema:
        feat_imp_df = model_service.raw_joblib.get('feature_importances', None)
        
        # Global Feature Importances
        global_items: List[FeatureImportanceItem] = []
        if feat_imp_df is not None:
            for _, row in feat_imp_df.head(15).iterrows():
                f_name = str(row['Feature'])
                desc = FEATURE_DESCRIPTIONS.get(f_name, f"Model feature impact weight on price prediction")
                global_items.append(FeatureImportanceItem(
                    feature=f_name,
                    importance=round(float(row['Importance']), 5),
                    description=desc
                ))

        # Check if local explanation requested
        base_val_lakhs = None
        pred_val_lakhs = None
        net_impact_lakhs = None
        positive_drivers: Optional[List[LocalContributionItem]] = None
        negative_drivers: Optional[List[LocalContributionItem]] = None
        all_contributions: Optional[List[LocalContributionItem]] = None
        ai_explanation: Optional[str] = None

        if property_input is not None:
            input_dict = property_input.model_dump()
            sample_df = pd.DataFrame([input_dict])
            
            # Predict
            pred_val = float(self.pipeline.predict(sample_df)[0])
            pred_val_lakhs = round(pred_val, 2)
            
            # Transform
            X_trans = self.preprocessor.transform(sample_df)
            shap_vals = self.explainer.shap_values(X_trans)[0]
            base_val = float(self.explainer.expected_value[0])
            base_val_lakhs = round(base_val, 2)
            net_impact_lakhs = round(pred_val_lakhs - base_val_lakhs, 2)

            feature_names = get_transformed_feature_names(self.preprocessor, sample_df)
            contrib_map = dict(zip(feature_names, shap_vals))

            # Map into intuitive high-level property attributes
            group_definitions = [
                (
                    f"City Location ({input_dict['City']})",
                    f"{input_dict['City']}",
                    [k for k in feature_names if k.startswith('City_')]
                ),
                (
                    f"Locality Tier ({input_dict['Locality_Type']})",
                    f"{input_dict['Locality_Type']} Tier",
                    [k for k in feature_names if k.startswith('Locality_Type_')]
                ),
                (
                    "Super Built-up Area",
                    f"{input_dict['Super_Area_SqFt']:,.0f} SqFt",
                    ['Super_Area_SqFt', 'Area_Diff_SqFt', 'Area_per_BHK', 'Area_per_Bathroom']
                ),
                (
                    "Carpet Area & Layout Efficiency",
                    f"{input_dict['Carpet_Area_SqFt']:,.0f} SqFt ({(input_dict['Carpet_Area_SqFt']/input_dict['Super_Area_SqFt'])*100:.0f}% usable)",
                    ['Carpet_Area_SqFt', 'Carpet_Super_Ratio']
                ),
                (
                    "Property Age",
                    f"{input_dict['Age_of_Property']} Years Old",
                    ['Age_of_Property', 'Age_Group_Established_15-25', 'Age_Group_Modern_5-15', 'Age_Group_New_0-5', 'Age_Group_Older_25+']
                ),
                (
                    "Transit & City Proximity",
                    f"{input_dict['Distance_to_Metro_km']} km Metro / {input_dict['Distance_to_City_Center_km']} km Center",
                    ['Distance_to_Metro_km', 'Distance_to_City_Center_km', 'Total_Distance_km', 'Distance_Ratio']
                ),
                (
                    "Floor Level & Height",
                    f"Floor {input_dict['Floor_Number']} of {input_dict['Total_Floors']}",
                    ['Floor_Number', 'Total_Floors', 'Floor_Ratio', 'Is_Ground_Floor', 'Is_Top_Floor']
                ),
                (
                    "Layout & Unit Size",
                    f"{input_dict['BHK']} BHK / {input_dict['Bathrooms']} Bath",
                    ['BHK', 'Bathrooms', 'Bath_per_BHK', 'Total_Rooms']
                ),
                (
                    "Amenities & Security",
                    f"Parking: {'Yes' if input_dict['Parking'] else 'No'}, Lift: {'Yes' if input_dict['Lift_Available'] else 'No'}, Gated: {'Yes' if input_dict['Gated_Community'] else 'No'}",
                    ['Parking', 'Lift_Available', 'Gated_Community', 'Amenity_Score']
                ),
                (
                    "Property Type",
                    f"{input_dict['Property_Type']}",
                    [k for k in feature_names if k.startswith('Property_Type_')]
                ),
                (
                    "Furnishing Status",
                    f"{input_dict['Furnishing_Status']}",
                    [k for k in feature_names if k.startswith('Furnishing_Status_')]
                )
            ]

            all_contrib_list: List[LocalContributionItem] = []
            pos_list: List[LocalContributionItem] = []
            neg_list: List[LocalContributionItem] = []

            for title, raw_desc, feat_keys in group_definitions:
                c_val = sum(contrib_map.get(k, 0.0) for k in feat_keys)
                c_rounded = round(c_val, 2)
                item = LocalContributionItem(
                    feature=title,
                    raw_value=raw_desc,
                    contribution_lakhs=c_rounded,
                    is_positive=(c_rounded >= 0.0)
                )
                all_contrib_list.append(item)
                if c_rounded > 0.05:
                    pos_list.append(item)
                elif c_rounded < -0.05:
                    neg_list.append(item)

            pos_list.sort(key=lambda x: x.contribution_lakhs, reverse=True)
            neg_list.sort(key=lambda x: x.contribution_lakhs) # most negative first
            all_contrib_list.sort(key=lambda x: abs(x.contribution_lakhs), reverse=True)

            positive_drivers = pos_list
            negative_drivers = neg_list
            all_contributions = all_contrib_list

            # Construct factual AI explanation
            pos_phrases = [f"{p.feature} (+₹ {p.contribution_lakhs:.2f} L)" for p in pos_list[:3]]
            neg_phrases = [f"{n.feature} (₹ {n.contribution_lakhs:.2f} L)" for n in neg_list[:2]]

            pos_desc = ", ".join(pos_phrases) if pos_phrases else "standard baseline features"
            neg_desc = ", while " + ", ".join(neg_phrases) + " exert downward price pressure." if neg_phrases else "with no material structural discounts identified."

            sign = "+" if net_impact_lakhs >= 0 else ""
            ai_explanation = (
                f"Starting from the dataset baseline of ₹ {base_val_lakhs:.2f} Lakhs, the model predicts "
                f"a valuation of ₹ {pred_val_lakhs:.2f} Lakhs (net {sign}₹ {net_impact_lakhs:.2f} L). "
                f"The primary positive drivers are {pos_desc}, {neg_desc}"
            )

        return ExplainabilityResponseSchema(
            base_value_lakhs=base_val_lakhs,
            predicted_price_lakhs=pred_val_lakhs,
            net_impact_lakhs=net_impact_lakhs,
            positive_drivers=positive_drivers,
            negative_drivers=negative_drivers,
            all_contributions=all_contributions,
            ai_explanation=ai_explanation,
            feature_importances=global_items,
            shap_summary_plot_url="/visualizations/shap_summary_plot.png",
            actual_vs_predicted_plot_url="/visualizations/actual_vs_predicted.png",
            error_by_city_plot_url="/visualizations/error_by_city.png"
        )

explainability_service = ExplainabilityService()
