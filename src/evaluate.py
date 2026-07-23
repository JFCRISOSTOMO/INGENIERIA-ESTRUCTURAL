"""
evaluate.py

Cálculo de métricas (R², RMSE, MAE) sobre el conjunto de prueba, comparación
contra el método empírico normativo (Ley de Abrams / ACI 211.1), y verificación
de estabilidad de resultados con y sin filas duplicadas (marco VDS - pilar de
Estabilidad).

Proyecto: Predicción de la Resistencia a Compresión del Concreto mediante ML
Curso: Aplicaciones de IA en Estructuras
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Calcula R², RMSE y MAE de un modelo sobre el conjunto de prueba."""
    y_pred = model.predict(X_test)
    return {
        "Modelo": name,
        "R2": round(r2_score(y_test, y_pred), 3),
        "RMSE (MPa)": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "MAE (MPa)": round(mean_absolute_error(y_test, y_pred), 2),
    }


def _abrams(wc, A, B):
    """Modelo empírico de Abrams: f'c = A / B^(w/c)."""
    return A / (B ** wc)


def evaluate_aci_baseline(X_train, y_train, X_test, y_test) -> dict:
    """
    Ajusta la fórmula de Abrams (base del enfoque ACI 211.1) sobre el conjunto
    de entrenamiento — replicando cómo se calibraría en obra con ensayos
    locales — y la evalúa sobre el mismo conjunto de prueba que los modelos ML,
    para que la comparación sea justa.
    """
    wc_train = X_train["Water"] / X_train["Cement"]
    wc_test = X_test["Water"] / X_test["Cement"]

    popt, _ = curve_fit(_abrams, wc_train, y_train, p0=[100, 10], maxfev=10000)
    A_fit, B_fit = popt

    y_pred_aci = _abrams(wc_test, A_fit, B_fit)

    return {
        "Modelo": "Empírico (Abrams / ACI 211.1)",
        "R2": round(r2_score(y_test, y_pred_aci), 3),
        "RMSE (MPa)": round(np.sqrt(mean_squared_error(y_test, y_pred_aci)), 2),
        "MAE (MPa)": round(mean_absolute_error(y_test, y_pred_aci), 2),
        "A": round(A_fit, 2),
        "B": round(B_fit, 2),
    }


def rmse_reduction(results_df: pd.DataFrame,
                    ml_model_name: str,
                    baseline_name: str = "Empírico (Abrams / ACI 211.1)") -> float:
    """% de reducción de RMSE de un modelo ML frente a la línea base empírica."""
    rmse_ml = results_df.loc[results_df["Modelo"] == ml_model_name, "RMSE (MPa)"].values[0]
    rmse_base = results_df.loc[results_df["Modelo"] == baseline_name, "RMSE (MPa)"].values[0]
    return round((rmse_base - rmse_ml) / rmse_base * 100, 1)


def stability_check(df: pd.DataFrame, train_fn, split_fn) -> pd.DataFrame:
    """
    Verifica la estabilidad de las métricas del mejor modelo (marco VDS)
    entrenando con y sin las 25 filas duplicadas identificadas en el EDA.

    Parameters
    ----------
    train_fn : función de entrenamiento (p.ej. train.train_xgboost)
    split_fn : función de split (p.ej. train.split_data)
    """
    rows = []
    for label, data in [("Con duplicados", df), ("Sin duplicados", df.drop_duplicates())]:
        X_train, X_test, y_train, y_test = split_fn(data)
        model = train_fn(X_train, y_train, verbose=False)
        result = evaluate_model(model, X_test, y_test, label)
        rows.append(result)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from data_loader import get_clean_data
    from train import split_data, train_random_forest, train_xgboost

    df = get_clean_data("../data/raw/Concrete_Data.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    results = [
        evaluate_model(rf_model, X_test, y_test, "Random Forest"),
        evaluate_model(xgb_model, X_test, y_test, "XGBoost"),
        evaluate_aci_baseline(X_train, y_train, X_test, y_test),
    ]
    results_df = pd.DataFrame(results)
    print(results_df)

    reduction = rmse_reduction(results_df, "XGBoost")
    print(f"\nReducción de RMSE de XGBoost frente al método empírico: {reduction}%")
