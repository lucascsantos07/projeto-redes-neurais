import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CLASS_TO_INDEX


def build_preprocessor(numeric_features, categorical_features):
    """
    Cria o pré-processador das variáveis preditoras.

    Numéricas: StandardScaler.
    Categóricas: OneHotEncoder.
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def preprocess_features(splits, numeric_features, categorical_features):
    """
    Ajusta o pré-processador no treino e transforma
    treino, validação e teste.
    """
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    # O fit acontece exclusivamente no treino
    X_train = preprocessor.fit_transform(splits["X_train"])

    # Validação e teste recebem somente transform
    X_val = preprocessor.transform(splits["X_val"])
    X_test = preprocessor.transform(splits["X_test"])

    feature_names = preprocessor.get_feature_names_out()

    return {
        "X_train": np.asarray(X_train, dtype=np.float32),
        "X_val": np.asarray(X_val, dtype=np.float32),
        "X_test": np.asarray(X_test, dtype=np.float32),
        "preprocessor": preprocessor,
        "feature_names": feature_names.tolist(),
    }


def scale_regression_target(splits):
    """
    Padroniza o alvo da regressão usando somente o treino.

    Retorna os alvos padronizados e o scaler,
    necessário para reverter as previsões posteriormente.
    """
    target_scaler = StandardScaler()

    y_train = target_scaler.fit_transform(
        splits["y_train"].to_numpy().reshape(-1, 1)
    )

    y_val = target_scaler.transform(
        splits["y_val"].to_numpy().reshape(-1, 1)
    )

    y_test = target_scaler.transform(
        splits["y_test"].to_numpy().reshape(-1, 1)
    )

    return {
        "y_train": y_train.astype(np.float32),
        "y_val": y_val.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "target_scaler": target_scaler,
    }


def encode_classification_target(splits):
    """
    Converte os rótulos textuais em números inteiros.

    Low      -> 0
    Moderate -> 1
    High     -> 2
    """
    encoded_targets = {}

    for split_name in ["train", "val", "test"]:
        y = splits[f"y_{split_name}"]

        encoded = y.map(CLASS_TO_INDEX)

        if encoded.isnull().any():
            unknown_classes = y[encoded.isnull()].unique()
            raise ValueError(
                f"Classes não reconhecidas: {unknown_classes}"
            )

        encoded_targets[f"y_{split_name}"] = (
            encoded.to_numpy(dtype=np.int64)
        )

    return encoded_targets
