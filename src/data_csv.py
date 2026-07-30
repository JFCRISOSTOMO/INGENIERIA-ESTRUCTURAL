"""Carga y limpieza del dataset UCI Concrete Compressive Strength."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Concrete_Data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COLUMN_NAMES = [
    "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
    "Superplasticizer", "CoarseAggregate", "FineAggregate",
    "Age", "Strength",
]


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = COLUMN_NAMES
    return df


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas totalmente duplicadas (25 de 1030, ver notebooks/01_eda_prueba.ipynb).

    Es la única eliminación de filas del proyecto. No se filtran outliers ni
    edades largas: corresponden a ensayos normados válidos, según la decisión
    tomada en el EDA.
    """
    return df.drop_duplicates().reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=random_state)


def save_processed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


def run_pipeline() -> None:
    df_raw = load_raw_data()
    n_raw = len(df_raw)

    df_clean = remove_exact_duplicates(df_raw)
    n_clean = len(df_clean)

    train_df, test_df = split_data(df_clean)
    save_processed(train_df, test_df)

    print(f"Filas crudas: {n_raw}")
    print(f"Filas tras eliminar duplicados exactos: {n_clean} ({n_raw - n_clean} eliminadas)")
    print(f"Train: {len(train_df)} filas | Test: {len(test_df)} filas")
    print(f"Guardado en {PROCESSED_DIR}")


if __name__ == "__main__":
    run_pipeline()
