import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.feature_engineering import RealEstateFeatureEngineer

def split_data(df: pd.DataFrame, target_col: str = 'Price_INR_Lakhs', test_size: float = 0.20, random_state: int = 42):
    """
    Performs train/test split BEFORE fitting preprocessors to prevent data leakage.
    Drop ID column if present.
    """
    X = df.drop(columns=[target_col])
    if 'Property_ID' in X.columns:
        X = X.drop(columns=['Property_ID'])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(sample_X: pd.DataFrame):
    """
    Builds a ColumnTransformer dynamically based on feature types after Feature Engineering.
    """
    # First apply feature engineering transformer to sample data to identify resulting feature names & types
    fe = RealEstateFeatureEngineer()
    fe_sample = fe.fit_transform(sample_X)
    
    categorical_cols = fe_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = fe_sample.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('feature_engineer', RealEstateFeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline, numerical_cols, categorical_cols
