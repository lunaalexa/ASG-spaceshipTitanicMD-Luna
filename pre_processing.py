"""
STEP 2: Preprocessing 
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def feature_engineering(df):
    """Apply comprehensive feature engineering"""
    df = df.copy()
    
    # Extract features from Cabin
    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown')
    df['Cabin_num'] = df['Cabin'].apply(lambda x: x.split('/')[1] if pd.notna(x) else -1).astype(float)
    df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown')
    
    # Extract group and individual from PassengerId
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['Group_size'] = df.groupby('Group')['Group'].transform('count')
    df['Solo'] = (df['Group_size'] == 1).astype(int)
    
    # Extract first and Last name
    df['FirstName'] = df['Name'].apply(lambda x: x.split()[0] if pd.notna(x) else 'Unknown')
    df['LastName'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notna(x) else 'Unknown')
    df['Family_size'] = df.groupby('LastName')['LastName'].transform('count')
    
    # Total spending features
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
    df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)
    
    # Spending ratios
    for col in spending_cols:
        df[f'{col}_ratio'] = df[col] / (df['TotalSpending'] + 1)
    
    # Age groups
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100], 
                              labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior'])
    df['Age_group'] = df['Age_group'].astype(str)
    
    # Missing value indicators
    df['Age_missing'] = df['Age'].isna().astype(int)
    df['CryoSleep_missing'] = df['CryoSleep'].isna().astype(int)
    
    return df

def preprocess():
    os.makedirs("artifacts", exist_ok=True)
    df = pd.read_csv("ingested/train.csv")

    df = feature_engineering(df)

    # Define features to use
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                         'Cabin_num', 'Group_size', 'Solo', 'Family_size', 'TotalSpending',
                         'HasSpending', 'NoSpending', 'Age_missing', 'CryoSleep_missing'] + \
                        [col for col in df.columns if '_ratio' in col]

    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')
    
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Select features
    feature_columns = categorical_features + numerical_features
    X = df[feature_columns]
    y = df['Transported'].astype(int)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns).reset_index(drop=True)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns).reset_index(drop=True)

    # save artifacts
    artifacts ={
        'scaler':scaler,
        'label_encoders':label_encoders,
        'feature_columns':feature_columns,
        'categorical_features':categorical_features,
        'numerical_features':numerical_features
    }
    joblib.dump(artifacts, "artifacts/preprocessor.pkl")

    # concat
    train_scaled = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    val_scaled = pd.concat([X_val_scaled, y_val.reset_index(drop=True)], axis=1)

    print(f"Preprocessing done. Artifacts saved to artifacts/preprocessor.pkl")
    return train_scaled, val_scaled


if __name__ == "__main__":
    preprocess()