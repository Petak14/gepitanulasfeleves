import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
import os

def main():
    
    # Adat betöltése
    df = pd.read_csv('C:/Users/gerol/Downloads/gepitanfeleves/GeneralEsportData.csv')
    print("First 5 records:")
    print(df.head())

    # Numerikus és nem numerikus oszlopok kiválasztása
    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = ['ReleaseDate', 'OfflineEarnings', 'PercentOffline', 'TotalPlayers', 'TotalTournaments']
    df_non_numeric = df.select_dtypes(exclude=[np.number])
    non_numeric_cols = df_non_numeric.columns.tolist()

    # Hiányzó értékek imputálása median-nal numerikus oszlopokban
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            med = df[col].median()
            df[col] = df[col].fillna(med)

    # Célozzuk meg a 'TotalEarnings' oszlopot regressziós célváltozóként
    y = df['TotalEarnings']
    X = df[['Game', 'Genre', 'ReleaseDate', 'OfflineEarnings', 'PercentOffline', 'TotalPlayers', 'TotalTournaments']]

    # Szétválasztjuk input változókat típus szerint
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Célváltozó logaritmikus transzformációja
    y_log = np.log1p(y)

    # Preprocessing pipeline numerikus és kategorikus változókra
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, non_numeric_cols)
    ])

    # Modell pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=355, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42))
    ])

    # Adatok felosztása train/test-re
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    print("Train set columns:", X_train.columns.tolist())
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", non_numeric_cols)

    # Modell tanítása
    model.fit(X_train, y_train)

    # Előrejelzés
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # visszatranszformálás eredeti skálára

    # Értékelés
    mse = mean_squared_error(np.expm1(y_test), y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(np.expm1(y_test), y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"A modell pontossága (R²): {r2 * 100:.2f}%")

if __name__ == "__main__":
    main()
