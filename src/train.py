"""
train.py

Entrenamiento de los modelos Random Forest y XGBoost con búsqueda de
hiperparámetros vía 5-fold Cross-Validation, sobre un split 80/20 fijo
(random_state=42), conforme al protocolo de validación definido en T2.

Proyecto: Predicción de la Resistencia a Compresión del Concreto mediante ML
Curso: Aplicaciones de IA en Estructuras
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
}

XGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
}


def split_data(df: pd.DataFrame, target: str = "CompressiveStrength"):
    """Separa features/target y aplica el holdout 80/20 con semilla fija."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def get_kfold():
    """5-fold CV con shuffle y semilla fija, usado solo para tuning."""
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def train_random_forest(X_train, y_train, verbose: bool = True):
    """Ajusta hiperparámetros de Random Forest vía GridSearchCV (5-fold)."""
    grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        RF_PARAM_GRID,
        cv=get_kfold(),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    if verbose:
        print("Random Forest — mejores hiperparámetros:", grid.best_params_)
    return grid.best_estimator_


def train_xgboost(X_train, y_train, verbose: bool = True):
    """Ajusta hiperparámetros de XGBoost vía GridSearchCV (5-fold)."""
    grid = GridSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        XGB_PARAM_GRID,
        cv=get_kfold(),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    if verbose:
        print("XGBoost — mejores hiperparámetros:", grid.best_params_)
    return grid.best_estimator_


def save_model(model, path: str):
    """Guarda un modelo entrenado en disco (formato joblib)."""
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


def load_model(path: str):
    """Carga un modelo previamente entrenado."""
    return joblib.load(path)


if __name__ == "__main__":
    from data_loader import get_clean_data

    df = get_clean_data("../data/raw/Concrete_Data.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    save_model(rf_model, "../results/random_forest_model.joblib")
    save_model(xgb_model, "../results/xgboost_model.joblib")
