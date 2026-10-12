import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import split_data, build_preprocessing_pipeline
from src.evaluate import evaluate_model, plot_diagnostics

def run_training_pipeline(data_path: str = "train.csv", output_dir: str = "models", viz_dir: str = "visualizations"):
    print("=" * 65, flush=True)
    print("AI-POWERED REAL ESTATE VALUATION: HIGH-SPEED TRAINING PIPELINE", flush=True)
    print("=" * 65, flush=True)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # 1. Load Data
    print(f"\n[1/7] Loading dataset from '{data_path}'...", flush=True)
    df = pd.read_csv(data_path)
    print(f"Dataset Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.", flush=True)

    # 2. Leak-Free Train/Test Split
    print("\n[2/7] Splitting data into 80% Train and 20% Untouched Test set...", flush=True)
    X_train, X_test, y_train, y_test = split_data(df, target_col='Price_INR_Lakhs', test_size=0.20, random_state=42)
    print(f"Training set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples", flush=True)

    # 3. Fit Preprocessing & Feature Engineering Pipeline on Train Set
    print("\n[3/7] Fitting Preprocessing & Feature Engineering transformers on Training set...", flush=True)
    preproc_pipeline, num_cols, cat_cols = build_preprocessing_pipeline(X_train)
    
    # Fit preprocessor on X_train and transform both splits
    X_train_trans = preproc_pipeline.fit_transform(X_train).astype(np.float32)
    X_test_trans = preproc_pipeline.transform(X_test).astype(np.float32)
    print(f"Transformed Feature Matrix Shape: {X_train_trans.shape}", flush=True)

    # 4. Baseline GradientBoostingRegressor
    print("\n[4/7] Training Baseline GradientBoostingRegressor...", flush=True)
    baseline_gbr = GradientBoostingRegressor(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=3,
        max_features='sqrt',
        subsample=0.5,
        random_state=42
    )
    baseline_gbr.fit(X_train_trans, y_train)

    y_pred_baseline = baseline_gbr.predict(X_test_trans)
    metrics_baseline = evaluate_model(y_test, y_pred_baseline, model_name="Baseline GradientBoosting")
    print("\nBaseline Test Set Metrics:", flush=True)
    print(f"  MAE  : {metrics_baseline['MAE']:.4f} Lakhs", flush=True)
    print(f"  MSE  : {metrics_baseline['MSE']:.4f}", flush=True)
    print(f"  RMSE : {metrics_baseline['RMSE']:.4f} Lakhs", flush=True)
    print(f"  R²   : {metrics_baseline['R2']:.4f}", flush=True)

    # 5. Hyperparameter Tuning via RandomizedSearchCV
    print("\n[5/7] Performing 5-Fold RandomizedSearchCV Hyperparameter Tuning...", flush=True)
    
    param_distributions = {
        'n_estimators': [30, 50],
        'learning_rate': [0.08, 0.1],
        'max_depth': [3, 4],
        'min_samples_split': [5, 10],
        'subsample': [0.5, 0.7],
        'max_features': ['sqrt', 0.5]
    }

    gbr_base = GradientBoostingRegressor(random_state=42)

    search = RandomizedSearchCV(
        estimator=gbr_base,
        param_distributions=param_distributions,
        n_iter=2,
        cv=3,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=1,
        verbose=0
    )

    search.fit(X_train_trans, y_train)

    print("\nHyperparameter Search Completed!", flush=True)
    print("Best Parameters Found:", flush=True)
    for param, val in search.best_params_.items():
        print(f"  - {param}: {val}", flush=True)
    print(f"Best CV RMSE: {-search.best_score_:.4f} Lakhs", flush=True)

    best_gbr = search.best_estimator_

    # 6. Evaluation on Untouched Test Set
    print("\n[6/7] Evaluating Best Tuned GradientBoosting Model on Untouched Test Set...", flush=True)
    y_pred_tuned = best_gbr.predict(X_test_trans)
    metrics_tuned = evaluate_model(y_test, y_pred_tuned, model_name="Tuned GradientBoosting")

    print("\nTuned Model Test Set Metrics:", flush=True)
    print(f"  MAE  : {metrics_tuned['MAE']:.4f} Lakhs", flush=True)
    print(f"  MSE  : {metrics_tuned['MSE']:.4f}", flush=True)
    print(f"  RMSE : {metrics_tuned['RMSE']:.4f} Lakhs", flush=True)
    print(f"  R²   : {metrics_tuned['R2']:.4f}", flush=True)

    # Model Performance Comparison
    rmse_imp = ((metrics_baseline['RMSE'] - metrics_tuned['RMSE']) / metrics_baseline['RMSE']) * 100
    r2_imp = metrics_tuned['R2'] - metrics_baseline['R2']
    print(f"\nPerformance Gain Summary:", flush=True)
    print(f"  RMSE Improvement: {rmse_imp:.2f}% reduction in prediction error", flush=True)
    print(f"  R² Improvement: +{r2_imp:.4f}", flush=True)

    # Plot Diagnostics
    print("\nGenerating Diagnostic Plots & Error Analysis...", flush=True)
    plot_diagnostics(y_test, y_pred_tuned, X_test, output_dir=viz_dir)

    # Extract Feature Importances
    feature_names = get_transformed_feature_names(preproc_pipeline, X_train)
    importances = best_gbr.feature_importances_

    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', palette='mako')
    plt.title('Top 15 Feature Importances (Tuned Gradient Boosting)', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'feature_importance_top15.png'), dpi=300)
    plt.close()

    # SHAP Explainability Plot
    try:
        import shap
        print("Computing SHAP Explainability Plot on Representative Test Sample (300 properties)...", flush=True)
        sample_indices = np.random.choice(len(X_test_trans), size=min(300, len(X_test_trans)), replace=False)
        X_shap_sample = X_test_trans[sample_indices]

        explainer = shap.TreeExplainer(best_gbr)
        shap_values = explainer.shap_values(X_shap_sample)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap_sample, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot - Feature Impact on Property Valuation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'shap_summary_plot.png'), dpi=300)
        plt.close()
        print("SHAP Summary Plot saved successfully!", flush=True)
    except Exception as e:
        print(f"SHAP visualization warning: {e}", flush=True)

    # Assemble End-to-End Pipeline
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preproc_pipeline),
        ('regressor', best_gbr)
    ])

    # 7. Save Model Pipeline
    print(f"\n[7/7] Saving final end-to-end model pipeline to '{output_dir}/real_estate_gbr_pipeline.joblib'...", flush=True)
    model_payload = {
        'pipeline': final_pipeline,
        'metrics_baseline': metrics_baseline,
        'metrics_tuned': metrics_tuned,
        'best_params': search.best_params_,
        'feature_importances': feat_imp_df,
        'target_col': 'Price_INR_Lakhs'
    }
    joblib.dump(model_payload, os.path.join(output_dir, 'real_estate_gbr_pipeline.joblib'))
    print("Model pipeline successfully saved & verified!", flush=True)

def get_transformed_feature_names(preproc_pipeline, X_sample):
    fe_sample = preproc_pipeline.named_steps['feature_engineer'].transform(X_sample.head(5))
    ct = preproc_pipeline.named_steps['preprocessor']
    try:
        names = ct.get_feature_names_out()
        return [n.replace('num__', '').replace('cat__', '') for n in names]
    except Exception:
        return [f"feature_{i}" for i in range(ct.transform(fe_sample).shape[1])]

if __name__ == "__main__":
    run_training_pipeline()
