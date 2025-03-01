import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler,OneHotEncoder, LabelEncoder

def min_max_scale(df: pd.DataFrame) -> pd.DataFrame:
    """ Applies Min-Max Scaling (0 to 1) to numerical columns."""
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """ Normalizes numerical columns (Z-score standardization)."""
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Applies One-Hot Encoding to categorical columns with more than 2 unique values and 
       Label Encoding to those with 2 or fewer unique values."""
    
    # One-Hot Encode categorical columns with more than 2 unique values
    one_hot_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in one_hot_cols:
        if df[col].nunique() > 2:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
        else:
            # Label Encode columns with 2 or fewer unique values
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    return df
