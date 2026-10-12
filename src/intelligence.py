import numpy as np
import pandas as pd

def calculate_valuation_score(row: pd.Series, predicted_price: float, city_stats: dict = None) -> float:
    """
    Computes a statistically justified Model-Based Property Valuation Score (0 to 100).
    Factors:
    - Area efficiency (Carpet to Super ratio)
    - Locality tier (Premium, Mid-Range, Affordable)
    - Proximity to Transit & City Center
    - Amenity completeness (Gated community, Lift, Parking)
    - Building age score
    """
    score = 50.0  # Base score
    
    # 1. Carpet-to-Super ratio efficiency (+/- 10 points)
    if 'Carpet_Area_SqFt' in row and 'Super_Area_SqFt' in row and row['Super_Area_SqFt'] > 0:
        ratio = row['Carpet_Area_SqFt'] / row['Super_Area_SqFt']
        if ratio >= 0.80:
            score += 10.0
        elif ratio >= 0.70:
            score += 5.0
        elif ratio < 0.60:
            score -= 5.0

    # 2. Locality Type (+/- 10 points)
    locality = str(row.get('Locality_Type', '')).lower()
    if 'premium' in locality:
        score += 12.0
    elif 'mid-range' in locality or 'midrange' in locality:
        score += 6.0

    # 3. Amenities (+/- 10 points)
    amenities = row.get('Parking', 0) + row.get('Lift_Available', 0) + row.get('Gated_Community', 0)
    score += (amenities * 3.5)

    # 4. Age penalty/bonus (+/- 10 points)
    age = row.get('Age_of_Property', 15)
    if age <= 3:
        score += 10.0
    elif age <= 10:
        score += 5.0
    elif age > 25:
        score -= 8.0

    # 5. Distance bonuses (+/- 10 points)
    metro_dist = row.get('Distance_to_Metro_km', 5.0)
    if metro_dist <= 1.5:
        score += 8.0
    elif metro_dist <= 3.5:
        score += 4.0

    city_dist = row.get('Distance_to_City_Center_km', 10.0)
    if city_dist <= 5.0:
        score += 8.0
    elif city_dist <= 10.0:
        score += 4.0

    # Clip score between 10 and 99
    final_score = float(np.clip(score, 10.0, 99.0))
    return round(final_score, 1)

def generate_intelligence_report(row: pd.Series, predicted_price: float, model_rmse: float = 25.0) -> dict:
    """
    Generates a comprehensive Real Estate Investment & Valuation Intelligence report.
    """
    valuation_score = calculate_valuation_score(row, predicted_price)
    
    # Valuation Range
    lower_bound = max(0.5, predicted_price - (0.84 * model_rmse))
    upper_bound = predicted_price + (0.84 * model_rmse)
    
    # Price per SqFt
    super_area = row.get('Super_Area_SqFt', 1000)
    carpet_area = row.get('Carpet_Area_SqFt', 750)
    
    price_per_sqft_super = (predicted_price * 100000) / super_area if super_area > 0 else 0
    price_per_sqft_carpet = (predicted_price * 100000) / carpet_area if carpet_area > 0 else 0

    # Valuation Assessment Tier
    if valuation_score >= 80:
        tier = "Prime Investment Asset"
        summary = "Highly attractive property with premium locality, strong transit proximity, and modern amenities."
    elif valuation_score >= 65:
        tier = "Strong Market Value"
        summary = "Well-positioned property matching market standards with solid fundamentals."
    elif valuation_score >= 50:
        tier = "Standard Value Asset"
        summary = "Average market listing. Standard feature set with moderate accessibility."
    else:
        tier = "Value / Discount Tier"
        summary = "Older property or further from transit/city center, reflected in accessible valuation."

    # Identify Key Value Drivers
    drivers = []
    if row.get('Locality_Type') == 'Premium':
        drivers.append("Location in Premium Locality Zone")
    if row.get('Gated_Community') == 1:
        drivers.append("Gated Community Security & Lifestyle")
    if row.get('Distance_to_Metro_km', 10) <= 2.0:
        drivers.append(f"Immediate Metro Accessibility ({row.get('Distance_to_Metro_km')} km)")
    if row.get('Age_of_Property', 20) <= 5:
        drivers.append(f"Modern Construction (Age: {row.get('Age_of_Property')} yrs)")
    if super_area >= 1800:
        drivers.append(f"Spacious Layout ({super_area:,.0f} SqFt Super Area)")

    discounts = []
    if row.get('Age_of_Property', 0) >= 25:
        discounts.append(f"Property Age ({row.get('Age_of_Property')} yrs)")
    if row.get('Distance_to_City_Center_km', 0) >= 15.0:
        discounts.append(f"Distance to City Center ({row.get('Distance_to_City_Center_km')} km)")
    if row.get('Lift_Available', 1) == 0:
        discounts.append("No Dedicated Elevator Service")

    return {
        "Predicted_Price_Lakhs": round(predicted_price, 2),
        "Predicted_Price_Crores": round(predicted_price / 100.0, 3),
        "Valuation_Lower_Lakhs": round(lower_bound, 2),
        "Valuation_Upper_Lakhs": round(upper_bound, 2),
        "Price_Per_SqFt_Super": round(price_per_sqft_super, 0),
        "Price_Per_SqFt_Carpet": round(price_per_sqft_carpet, 0),
        "Valuation_Score": valuation_score,
        "Valuation_Tier": tier,
        "Executive_Summary": summary,
        "Key_Value_Drivers": drivers,
        "Discount_Factors": discounts
    }
