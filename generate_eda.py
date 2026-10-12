import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data_path: str = "train.csv", output_dir: str = "visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Target Distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['Price_INR_Lakhs'], kde=True, color='#2b5c8f', bins=50)
    plt.title('Target Distribution: Price (₹ Lakhs)', fontsize=13, fontweight='bold')
    plt.xlabel('Price (₹ Lakhs)')
    plt.ylabel('Property Count')

    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(df['Price_INR_Lakhs']), kde=True, color='#27ae60', bins=50)
    plt.title('Log-Transformed Target Distribution', fontsize=13, fontweight='bold')
    plt.xlabel('Log(1 + Price)')
    plt.ylabel('Property Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=300)
    plt.close()

    # 2. Numerical Features Distribution & Outliers
    num_cols = ['Super_Area_SqFt', 'Carpet_Area_SqFt', 'BHK', 'Bathrooms', 'Floor_Number', 'Total_Floors', 'Age_of_Property', 'Distance_to_Metro_km', 'Distance_to_City_Center_km']
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(y=df[col], color='#4a90e2')
        plt.title(f'Boxplot: {col}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numerical_outliers_boxplots.png'), dpi=300)
    plt.close()

    # 3. Categorical Feature Distributions
    cat_cols = ['City', 'Locality_Type', 'Property_Type', 'Furnishing_Status']
    plt.figure(figsize=(16, 10))
    for i, col in enumerate(cat_cols, 1):
        plt.subplot(2, 2, i)
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, palette='viridis')
        plt.xticks(rotation=30)
        plt.title(f'Categorical Frequency: {col}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'categorical_distributions.png'), dpi=300)
    plt.close()

    # 4. Correlation Analysis
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

    # 5. Price vs Area Analysis
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='Super_Area_SqFt', y='Price_INR_Lakhs', hue='Property_Type', alpha=0.5)
    plt.title('Super Area vs Price (₹ Lakhs)', fontsize=12, fontweight='bold')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='Carpet_Area_SqFt', y='Price_INR_Lakhs', hue='Locality_Type', alpha=0.5)
    plt.title('Carpet Area vs Price (₹ Lakhs)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_vs_area.png'), dpi=300)
    plt.close()

    # 6. City Geographic Price Analysis
    plt.figure(figsize=(12, 6))
    city_order = df.groupby('City')['Price_INR_Lakhs'].median().sort_values(ascending=False).index
    sns.barplot(data=df, x='City', y='Price_INR_Lakhs', order=city_order, palette='plasma', estimator=np.median)
    plt.xticks(rotation=45)
    plt.title('Median Property Price by City (₹ Lakhs)', fontsize=14, fontweight='bold')
    plt.xlabel('City')
    plt.ylabel('Median Price (₹ Lakhs)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'city_median_prices.png'), dpi=300)
    plt.close()

    # 7. Locality Type vs Price
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Locality_Type', y='Price_INR_Lakhs', palette='Set1')
    plt.title('Property Price Distribution by Locality Tier', fontsize=14, fontweight='bold')
    plt.xlabel('Locality Type')
    plt.ylabel('Price (₹ Lakhs)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'locality_vs_price.png'), dpi=300)
    plt.close()

    print(f"EDA Visualizations generated and saved in '{output_dir}/'")

if __name__ == "__main__":
    perform_eda()
