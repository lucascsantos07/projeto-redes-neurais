
import numpy as np
import torch

from src.models import MLP


def load_model(checkpoint, device="cpu"):
    config = checkpoint["model_config"]

    model = MLP(
        in_dim=checkpoint["input_dim"],
        out_dim=checkpoint["output_dim"],
        hidden=tuple(config["hidden"]),
        activation=config["activation"],
        dropout=config.get("dropout", 0.0),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def predict_regression(model, checkpoint, X_new):
    """
    Realiza previsões de regressão a partir de dados originais.

    Retorna as previsões na escala original do alvo.
    """

    # 1. Aplicar o pré-processamento das variáveis de entrada
    preprocessor = checkpoint["preprocessor"]
    X_processed = preprocessor.transform(X_new)

    # 2. Converter os dados para tensor
    device = next(model.parameters()).device

    X_tensor = torch.as_tensor(
        np.asarray(X_processed, dtype=np.float32),
        device=device,
    )

    # 3. Realizar a previsão
    model.eval()

    with torch.inference_mode():
        predictions_scaled = model(X_tensor).cpu().numpy()

    # 4. Retornar à escala original
    target_scaler = checkpoint["target_scaler"]
    predictions = target_scaler.inverse_transform(predictions_scaled)

    return predictions.ravel()


def predict_classification(model, checkpoint, X_new):
    """
    Realiza previsões de classificação multiclasse.

    Retorna as classes previstas e suas probabilidades.
    """

    # 1. Aplicar o pré-processamento das variáveis de entrada
    preprocessor = checkpoint["preprocessor"]
    X_processed = preprocessor.transform(X_new)

    # 2. Converter os dados para tensor
    device = next(model.parameters()).device

    X_tensor = torch.as_tensor(
        np.asarray(X_processed, dtype=np.float32),
        device=device,
    )

    # 3. Calcular as probabilidades
    model.eval()

    with torch.inference_mode():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)

    # 4. Selecionar a classe com maior probabilidade
    predicted_indices = probabilities.argmax(dim=1).cpu().numpy()

    classes = checkpoint["classes"]
    predictions = [classes[index] for index in predicted_indices]

    return predictions, probabilities.cpu().numpy()
