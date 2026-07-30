"""
interpretability.py

Análisis de interpretabilidad mediante SHAP sobre el modelo Random Forest,
incluyendo la importancia global de variables y su variación según el rango
de resistencia objetivo (bajo / medio / alto) — el diferenciador solicitado
por el docente frente a Yeh (1998) y Chou et al. (2011).

Proyecto: Predicción de la Resistencia a Compresión del Concreto mediante ML
Curso: Aplicaciones de IA en Estructuras
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

COLUMN_NAMES_ES = {
    "Cement": "Cemento - kg/m3",
    "BlastFurnaceSlag": "EscoreaAltoHorno - kg/m3",
    "FlyAsh": "CenizaVolante - kg/m3",
    "Water": "Agua - kg/m3",
    "Superplasticizer": "Superplastificante - kg/m3",
    "CoarseAggregate": "AgregadoGrueso - kg/m3",
    "FineAggregate": "AgregadoFino - kg/m3",
    "Age": "Edad - dias",
}


def compute_shap_values(model, X_test: pd.DataFrame):
    """Calcula los valores SHAP del modelo (pensado para modelos de árboles)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values


def global_importance(shap_values, X_test: pd.DataFrame) -> pd.DataFrame:
    """Importancia global: media del valor absoluto de SHAP por variable."""
    importances = pd.DataFrame({
        "Variable": [COLUMN_NAMES_ES.get(c, c) for c in X_test.columns],
        "Importancia SHAP (media |valor|)": np.abs(shap_values).mean(axis=0)
    }).sort_values("Importancia SHAP (media |valor|)", ascending=False).reset_index(drop=True)
    return importances


def importance_by_strength_range(model, X_test: pd.DataFrame, y_test: pd.Series,
                                   low_q: float = 0.33, high_q: float = 0.66) -> pd.DataFrame:
    """
    Compara la importancia SHAP de cada variable entre mezclas de baja,
    media y alta resistencia (terciles de y_test). Responde directamente
    la pregunta del docente: "¿qué variables priorizar según la resistencia
    objetivo?"
    """
    low_cut = y_test.quantile(low_q)
    high_cut = y_test.quantile(high_q)

    ranges = {
        "Baja resistencia": y_test <= low_cut,
        "Media resistencia": (y_test > low_cut) & (y_test <= high_cut),
        "Alta resistencia": y_test > high_cut,
    }

    explainer = shap.TreeExplainer(model)
    rows = []
    for range_name, mask in ranges.items():
        X_subset = X_test[mask]
        if len(X_subset) == 0:
            continue
        shap_vals = explainer.shap_values(X_subset)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        for col, val in zip(X_test.columns, mean_abs):
            rows.append({
                "Rango de resistencia": range_name,
                "Variable": COLUMN_NAMES_ES.get(col, col),
                "Importancia SHAP": round(val, 3),
                "n": mask.sum(),
            })

    result = pd.DataFrame(rows)
    return result.pivot(index="Variable", columns="Rango de resistencia",
                         values="Importancia SHAP")[
        ["Baja resistencia", "Media resistencia", "Alta resistencia"]
    ].sort_values("Alta resistencia", ascending=False)


def plot_summary(shap_values, X_test: pd.DataFrame, save_path: str = None):
    """Genera el gráfico resumen de SHAP (beeswarm plot)."""
    X_display = X_test.rename(columns=COLUMN_NAMES_ES)
    shap.summary_plot(shap_values, X_display, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(".")
    from data_loader import get_clean_data
    from train import split_data, train_random_forest

    os.makedirs("../docs/figures", exist_ok=True)

    df = get_clean_data("../data/raw/Concrete_Data.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    rf_model = train_random_forest(X_train, y_train)

    shap_values = compute_shap_values(rf_model, X_test)

    print("\n=== Importancia global ===")
    print(global_importance(shap_values, X_test))

    print("\n=== Importancia por rango de resistencia ===")
    print(importance_by_strength_range(rf_model, X_test, y_test))

    plot_summary(shap_values, X_test, save_path="../docs/figures/shap_summary.png")
