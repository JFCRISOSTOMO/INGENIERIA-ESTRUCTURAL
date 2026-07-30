"""
main.py

Pipeline completo del proyecto — carga, entrenamiento, evaluación,
comparación empírica e interpretabilidad — en un solo script ejecutable.

Uso:
    cd src/
    python main.py

Proyecto: Predicción de la Resistencia a Compresión del Concreto mediante ML
Curso: Aplicaciones de IA en Estructuras
"""

import os
import pandas as pd

from data_loader import get_clean_data
from train import split_data, train_random_forest, train_xgboost, save_model
from evaluate import evaluate_model, evaluate_aci_baseline, rmse_reduction, stability_check
from interpretability import (
    compute_shap_values, global_importance, importance_by_strength_range, plot_summary
)

DATA_PATH = "../data/raw/Concrete_Data.csv"
FIGURES_DIR = "../docs/figures"
RESULTS_DIR = "../results"


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("1. CARGA DE DATOS")
    print("=" * 60)
    df = get_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas")

    print("\n" + "=" * 60)
    print("2. ENTRENAMIENTO (5-fold CV para hiperparámetros)")
    print("=" * 60)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    save_model(rf_model, f"{RESULTS_DIR}/random_forest_model.joblib")
    save_model(xgb_model, f"{RESULTS_DIR}/xgboost_model.joblib")

    print("\n" + "=" * 60)
    print("3. EVALUACIÓN SOBRE CONJUNTO DE PRUEBA")
    print("=" * 60)
    results = [
        evaluate_model(rf_model, X_test, y_test, "Random Forest"),
        evaluate_model(xgb_model, X_test, y_test, "XGBoost"),
        evaluate_aci_baseline(X_train, y_train, X_test, y_test),
    ]
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{RESULTS_DIR}/metrics_summary.csv", index=False)

    best_model_name = results_df.loc[results_df["RMSE (MPa)"][:2].idxmin(), "Modelo"]
    reduction = rmse_reduction(results_df, best_model_name)
    print(f"\nMejor modelo: {best_model_name}")
    print(f"Reducción de RMSE frente al método empírico ACI 211.1: {reduction}%")

    print("\n" + "=" * 60)
    print("4. VERIFICACIÓN DE ESTABILIDAD (con / sin duplicados)")
    print("=" * 60)
    stability_df = stability_check(df, train_xgboost, split_data)
    print(stability_df.to_string(index=False))
    stability_df.to_csv(f"{RESULTS_DIR}/stability_check.csv", index=False)

    print("\n" + "=" * 60)
    print("5. INTERPRETABILIDAD (SHAP) — Random Forest")
    print("=" * 60)
    shap_values = compute_shap_values(rf_model, X_test)

    importance_df = global_importance(shap_values, X_test)
    print("\nImportancia global de variables:")
    print(importance_df.to_string(index=False))
    importance_df.to_csv(f"{RESULTS_DIR}/shap_global_importance.csv", index=False)

    by_range_df = importance_by_strength_range(rf_model, X_test, y_test)
    print("\nImportancia por rango de resistencia:")
    print(by_range_df.to_string())
    by_range_df.to_csv(f"{RESULTS_DIR}/shap_by_strength_range.csv")

    plot_summary(shap_values, X_test, save_path=f"{FIGURES_DIR}/shap_summary.png")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO. Resultados guardados en:", RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
