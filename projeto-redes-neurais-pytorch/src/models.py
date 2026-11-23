import torch.nn as nn


class MLP(nn.Module):
    """
    Rede Neural Multilayer Perceptron parametrizável.

    Args:
        in_dim: Quantidade de features de entrada.
        out_dim: Quantidade de neurônios na saída.
        hidden: Tupla com a quantidade de neurônios
                em cada camada oculta.
        activation: Nome da função de ativação.
        dropout: Taxa de dropout entre 0 e 1.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden=(32, 32),
        activation="ReLU",
        dropout=0.0,
    ):
        super().__init__()

        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(
                "in_dim e out_dim devem ser positivos."
            )

        if not hidden or any(neurons <= 0 for neurons in hidden):
            raise ValueError(
                "hidden deve conter neurônios positivos."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout deve estar entre 0 e 1."
            )

        if not hasattr(nn, activation):
            raise ValueError(
                f"Ativação '{activation}' não encontrada em torch.nn."
            )

        activation_class = getattr(nn, activation)

        if not isinstance(activation_class, type) or not issubclass(
            activation_class, nn.Module
        ):
            raise ValueError(
                f"'{activation}' não é uma classe de ativação válida."
            )

        layers = []
        current_dim = in_dim

        # Construção das camadas ocultas
        for neurons in hidden:
            layers.append(
                nn.Linear(current_dim, neurons)
            )

            layers.append(activation_class())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = neurons

        # Camada de saída: sem ativação
        layers.append(
            nn.Linear(current_dim, out_dim)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
