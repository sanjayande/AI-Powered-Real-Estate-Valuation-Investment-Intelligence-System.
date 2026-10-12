import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Computes regression evaluation metrics: MAE, MSE, RMSE, R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }
    return metrics

def plot_diagnostics(y_true, y_pred, df_test, output_dir="visualizations"):
    """
    Generates regression diagnostic plots and price error analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_true - y_pred
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, color='#2b5c8f')
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.title('Actual vs Predicted Property Prices (Lakhs)', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Price (₹ Lakhs)', fontsize=12)
    plt.ylabel('Predicted Price (₹ Lakhs)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    # 2. Residual Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3, color='#e74c3c')
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.title('Residual Plot (Predicted Price vs Residuals)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Price (₹ Lakhs)', fontsize=12)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'), dpi=300)
    plt.close()
    
    # 3. Residual Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='#27ae60', bins=50)
    plt.axvline(0, color='black', linestyle='--', lw=1.5)
    plt.title('Residual Distribution Error Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Residual Error (₹ Lakhs)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_distribution.png'), dpi=300)
    plt.close()

    # 4. Error Analysis across Cities & Property Types
    temp_df = df_test.copy()
    temp_df['Actual_Price'] = y_true
    temp_df['Predicted_Price'] = y_pred
    temp_df['Abs_Error'] = np.abs(residuals)
    temp_df['Error_Percentage'] = (temp_df['Abs_Error'] / temp_df['Actual_Price']) * 100
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=temp_df, x='City', y='Abs_Error', palette='Set2')
    plt.xticks(rotation=45)
    plt.title('Absolute Valuation Error Distribution by City', fontsize=14, fontweight='bold')
    plt.xlabel('City', fontsize=12)
    plt.ylabel('Absolute Error (₹ Lakhs)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_by_city.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=temp_df, x='Property_Type', y='Abs_Error', palette='viridis')
    plt.title('Absolute Valuation Error Distribution by Property Type', fontsize=14, fontweight='bold')
    plt.xlabel('Property Type', fontsize=12)
    plt.ylabel('Absolute Error (₹ Lakhs)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_by_property_type.png'), dpi=300)
    plt.close()
    
    return temp_df
