import copy

import torch


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    epochs=200,
    patience=20,
    min_delta=1e-6,
):
    """
    Treina uma rede neural e utiliza validação
    para selecionar os melhores pesos (Early Stopping).

    Args:
        model: Modelo PyTorch.
        train_loader: DataLoader de treinamento.
        val_loader: DataLoader de validação.
        loss_fn: Função de perda.
        optimizer: Otimizador.
        epochs: Número máximo de épocas.
        patience: Quantidade de épocas sem melhora permitida.
        min_delta: Melhora mínima para considerar progresso.

    Returns:
        Dicionário com histórico e informações do treinamento.
    """

    if epochs <= 0 or patience <= 0:
        raise ValueError(
            "epochs e patience devem ser positivos."
        )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    waiting = 0
    best_state = None

    for epoch in range(epochs):

        # -------------------------
        # Etapa de treinamento
        # -------------------------
        model.train()

        train_loss_sum = 0.0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = loss_fn(predictions, y_batch)

            loss.backward()

            optimizer.step()

            train_loss_sum += loss.item() * len(X_batch)

        train_loss = (
            train_loss_sum / len(train_loader.dataset)
        )

        # -------------------------
        # Etapa de validação
        # -------------------------
        model.eval()

        val_loss_sum = 0.0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                predictions = model(X_batch)

                loss = loss_fn(predictions, y_batch)

                val_loss_sum += loss.item() * len(X_batch)

        val_loss = (
            val_loss_sum / len(val_loader.dataset)
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # -------------------------
        # Early Stopping
        # -------------------------
        if val_loss < best_val_loss - min_delta:

            best_val_loss = val_loss
            best_epoch = epoch + 1
            waiting = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

        else:
            waiting += 1

        print(
            f"Época [{epoch + 1}/{epochs}] "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f}"
        )

        if waiting >= patience:
            print(
                f"\nEarly Stopping na época {epoch + 1}."
            )
            break

    # Restaura os melhores pesos encontrados
    if best_state is None:
        raise RuntimeError(
            "Não foi possível encontrar um estado válido do modelo."
        )

    model.load_state_dict(best_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
    }
