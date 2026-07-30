"""
data_loader.py

Carga y limpieza del dataset Concrete Compressive Strength (UCI).
Proyecto: Predicción de la Resistencia a Compresión del Concreto mediante ML
Curso: Aplicaciones de IA en Estructuras

El archivo original de UCI usa punto y coma (;) como separador y encabezados
largos con unidades entre paréntesis. Este módulo estandariza ambos.
"""

import pandas as pd

COLUMN_NAMES = [
    "Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
    "CoarseAggregate", "FineAggregate", "Age", "CompressiveStrength",
]

COLUMN_NAMES_ES = {
    "Cement": "Cemento - kg/m3",
    "BlastFurnaceSlag": "EscoreaAltoHorno - kg/m3",
    "FlyAsh": "CenizaVolante - kg/m3",
    "Water": "Agua - kg/m3",
    "Superplasticizer": "Superplastificante - kg/m3",
    "CoarseAggregate": "AgregadoGrueso - kg/m3",
    "FineAggregate": "AgregadoFino - kg/m3",
    "Age": "Edad - dias",
    "CompressiveStrength": "ResistenciaCompresion - MPa",
}


def load_raw_data(path: str = "../data/raw/Concrete_Data.csv") -> pd.DataFrame:
    """
    Carga el dataset original de UCI y estandariza los nombres de columnas.

    Parameters
    ----------
    path : str
        Ruta al archivo Concrete_Data.csv (formato UCI, separador ';').

    Returns
    -------
    pd.DataFrame
        Dataset con 1030 filas y 9 columnas, nombres de columna estandarizados.
    """
    df = pd.read_csv(path, sep=";")
    df.columns = COLUMN_NAMES
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Verifica nulos y duplicados. No elimina nada por defecto — deja la
    decisión de tratamiento a la etapa de modelamiento (ver train.py),
    conforme al pilar de Estabilidad del marco VDS (evaluar con/sin duplicados).

    Returns
    -------
    dict con conteo de nulos totales y filas duplicadas.
    """
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_nulls": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
    }


def get_clean_data(path: str = "../data/raw/Concrete_Data.csv",
                    drop_duplicates: bool = False) -> pd.DataFrame:
    """
    Punto de entrada principal: carga + reporte de calidad + (opcional) dedup.

    Parameters
    ----------
    drop_duplicates : bool
        Si True, elimina las 25 filas duplicadas identificadas en el EDA (T2).
        Se deja en False por defecto para reproducir el pipeline reportado en
        el informe; se usa True únicamente en el análisis de sensibilidad de
        estabilidad (ver evaluate.py::stability_check).
    """
    df = load_raw_data(path)
    quality = check_data_quality(df)
    print(f"Dataset cargado: {quality['n_rows']} filas, {quality['n_cols']} columnas")
    print(f"Valores nulos: {quality['n_nulls']} | Filas duplicadas: {quality['n_duplicates']}")

    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Duplicados eliminados. Filas resultantes: {len(df)}")

    return df


if __name__ == "__main__":
    df = get_clean_data()
    print(df.describe().T)
