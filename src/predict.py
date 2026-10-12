import os
import joblib
import pandas as pd
from src.intelligence import generate_intelligence_report

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'real_estate_gbr_pipeline.joblib')

class RealEstatePredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model pipeline not found at {model_path}. Run training pipeline first.")
        
        saved_data = joblib.load(model_path)
        self.pipeline = saved_data['pipeline']
        self.metrics = saved_data.get('metrics', {})
        self.model_rmse = self.metrics.get('RMSE', 25.0)

    def predict(self, input_df: pd.DataFrame) -> pd.Series:
        """
        Predict property price in ₹ Lakhs for a DataFrame of inputs.
        """
        preds = self.pipeline.predict(input_df)
        return pd.Series(preds, index=input_df.index)

    def predict_with_intelligence(self, single_row: pd.Series or dict) -> dict:
        """
        Predicts price and generates intelligence report for a single property.
        """
        if isinstance(single_row, dict):
            df_input = pd.DataFrame([single_row])
            row_series = pd.Series(single_row)
        else:
            df_input = pd.DataFrame([single_row.to_dict()])
            row_series = single_row

        predicted_price = float(self.pipeline.predict(df_input)[0])
        report = generate_intelligence_report(row_series, predicted_price, self.model_rmse)
        return report
