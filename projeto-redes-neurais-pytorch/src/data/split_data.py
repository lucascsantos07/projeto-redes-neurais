from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)


def split_dataset(X, y, stratified=False):
    """
    Divide os dados em treino, validação e teste.

    O conjunto de teste é separado primeiro.
    A validação é retirada somente do conjunto de treino.

    Args:
        X: Variáveis preditoras.
        y: Variável alvo.
        stratified: Se True, preserva a proporção das classes.

    Returns:
        Dicionário com os seis conjuntos.
    """
    stratify_target = y if stratified else None

    # Primeira divisão: treino completo e teste
    X_train_full, X_test, y_train_full, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify_target,
        )
    )

    # Segunda divisão: treino e validação
    stratify_train = y_train_full if stratified else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_train,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def split_regression_data(df, target):
    """Realiza a divisão dos dados de regressão."""
    X = df.drop(columns=[target])
    y = df[target]

    return split_dataset(X, y, stratified=False)


def split_classification_data(df, target):
    """Realiza a divisão estratificada da classificação."""
    X = df.drop(columns=[target])
    y = df[target]

    return split_dataset(X, y, stratified=True)
