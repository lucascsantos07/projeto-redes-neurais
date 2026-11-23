import torch

from torch.utils.data import TensorDataset, DataLoader


def create_tensor_datasets(features, targets, task):
    """
    Converte os dados pré-processados em TensorDataset.

    Args:
        features: Dicionário com X_train, X_val e X_test.
        targets: Dicionário com y_train, y_val e y_test.
        task: 'regression' ou 'classification'.

    Returns:
        Dicionário com os datasets de treino, validação e teste.
    """
    if task not in {"regression", "classification"}:
        raise ValueError(
            "A tarefa deve ser 'regression' ou 'classification'."
        )

    datasets = {}

    for split in ["train", "val", "test"]:
        X = torch.tensor(
            features[f"X_{split}"],
            dtype=torch.float32,
        )

        if task == "regression":
            y = torch.tensor(
                targets[f"y_{split}"],
                dtype=torch.float32,
            )
        else:
            y = torch.tensor(
                targets[f"y_{split}"],
                dtype=torch.long,
            )

        datasets[split] = TensorDataset(X, y)

    return datasets


def create_dataloaders(
    datasets,
    batch_size=64,
    seed=42,
):
    """
    Cria DataLoaders para treino, validação e teste.

    Somente o DataLoader de treino embaralha os dados.
    """
    if batch_size <= 0:
        raise ValueError("O batch_size deve ser positivo.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=False,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        ),
    }

    return dataloaders
