import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def build_preprocessor(train_path: str):
    df = pd.read_csv(train_path)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(['Survived'], errors='ignore').tolist()

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    X = df.drop(columns=['Survived'])
    preprocessor.fit(X)
    
    os.makedirs('../models', exist_ok=True)
    joblib.dump(preprocessor, '../models/preprocessor.pkl')
    return preprocessor, cat_cols, num_cols