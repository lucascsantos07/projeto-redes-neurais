import pandas as pd

from src.config import (
    REGRESSION_DATA_PATH,
    CLASSIFICATION_DATA_PATH,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
)


def load_csv(file_path):
    """Carrega um arquivo CSV e retorna um DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {file_path}"
        )

    return pd.read_csv(file_path)


def validate_dataset(df, target):
    """Verifica se o dataset possui alvo e dados válidos."""
    if target not in df.columns:
        raise ValueError(
            f"A coluna alvo '{target}' não foi encontrada."
        )

    if df.empty:
        raise ValueError("O dataset está vazio.")

    if df.isnull().values.any():
        raise ValueError(
            "O dataset possui valores ausentes."
        )


def load_regression_data():
    """Carrega e valida o dataset de regressão."""
    df = load_csv(REGRESSION_DATA_PATH)
    validate_dataset(df, REGRESSION_TARGET)
    return df


def load_classification_data():
    """Carrega e valida o dataset de classificação."""
    df = load_csv(CLASSIFICATION_DATA_PATH)
    validate_dataset(df, CLASSIFICATION_TARGET)
    return df
