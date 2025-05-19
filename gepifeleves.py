import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv('GeneralEsportData.csv')
    return df

@st.cache_resource(show_spinner=True)
def train_model(df):
    # Hiányzó értékek kezelése
    numeric_cols = ['ReleaseDate', 'OfflineEarnings', 'PercentOffline', 'TotalPlayers', 'TotalTournaments']
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    y_log = np.log1p(df['TotalEarnings'])
    X = df[['Game', 'Genre', 'ReleaseDate', 'OfflineEarnings', 'PercentOffline', 'TotalPlayers', 'TotalTournaments']]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

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

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=355, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42))
    ])

    model.fit(X, y_log)
    return model

def main():
    st.title("Esport Total Earnings prediction")

    df = load_data()
    model = train_model(df)

    # UI inputok
    game = st.selectbox("Válaszd ki a játékot:", options=df['Game'].unique())
    genre = st.selectbox("Válaszd ki a műfajt:", options=df['Genre'].unique())
    release_date = st.number_input("Release Date (év)", min_value=1980, max_value=2030, value=2010)
    offline_earnings = st.number_input("Offline Earnings", min_value=0.0, value=0.0, step=1000.0)
    percent_offline = st.slider("Percent Offline (%)", min_value=0.0, max_value=100.0, value=50.0)
    total_players = st.number_input("Total Players", min_value=0, value=1000, step=100)
    total_tournaments = st.number_input("Total Tournaments", min_value=0, value=50, step=1)

    input_df = pd.DataFrame({
        'Game': [game],
        'Genre': [genre],
        'ReleaseDate': [release_date],
        'OfflineEarnings': [offline_earnings],
        'PercentOffline': [percent_offline],
        'TotalPlayers': [total_players],
        'TotalTournaments': [total_tournaments]
    })

    if st.button("Előrejelzés"):
        pred_log = model.predict(input_df)[0]
        pred = np.expm1(pred_log)
        st.success(f"Becsült Total Earnings: ${pred:,.2f}")

if __name__ == "__main__":
    main()
