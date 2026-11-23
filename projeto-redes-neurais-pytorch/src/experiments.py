import time

import numpy as np
import torch
import torch.nn as nn
from itertools import product

from torch.optim import (
    SGD,
    RMSprop,
    Adam,
    AdamW,
)

import copy

import pandas as pd

from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error

from src.models import MLP
from src.train import train_model
from src.data.torch_data import create_dataloaders
from src.utils.reproducibility import set_seed

def build_optimizer(model, optimizer_name, lr, weight_decay=0.0):

    name = optimizer_name.lower()

    if name == "sgd":
        return SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif name == "sgd_momentum":
        return SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )

    elif name == "rmsprop":
        return RMSprop(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif name == "adam":
        return Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif name == "adamw":
        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    else:
        raise ValueError(
            f"Otimizador não reconhecido: {optimizer_name}"
        )



def evaluate_model(model, dataloader, task, target_scaler=None):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in dataloader:
            predictions = model(X)

            if task == "regression":
                y_true.extend(y.cpu().numpy().reshape(-1))
                y_pred.extend(predictions.cpu().numpy().reshape(-1))

            elif task == "classification":
                predicted_classes = torch.argmax(predictions, dim=1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(predicted_classes.cpu().numpy())

            else:
                raise ValueError("Tarefa inválida.")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if task == "regression":
        if target_scaler is None:
            raise ValueError(
                "É necessário fornecer target_scaler para avaliar a regressão."
            )

        y_true = target_scaler.inverse_transform(
            y_true.reshape(-1, 1)
        ).reshape(-1)

        y_pred = target_scaler.inverse_transform(
            y_pred.reshape(-1, 1)
        ).reshape(-1)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return {
            "rmse": rmse,
            "mae": mae,
        }

    return {
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }

def run_experiment(
    datasets,
    task,
    config,
    seed=42,
    target_scaler=None,
):
    # 1. Reprodutibilidade
    set_seed(seed)

    # 2. Criar DataLoaders
    dataloaders = create_dataloaders(
        datasets,
        batch_size=config["batch_size"],
        seed=seed,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # 3. Definir dimensões da rede
    input_dim = datasets["train"].tensors[0].shape[1]

    if task == "regression":
        output_dim = 1

    elif task == "classification":
        output_dim = 3

    else:
        raise ValueError("Tarefa inválida.")

    loss_name = config.get(
        "loss",
        "mse" if task == "regression" else "cross_entropy"
    )

    loss_fn = build_loss_function(
        loss_name=loss_name,
        task=task,
        train_dataset=datasets["train"],
    )

    # 4. Instanciar modelo
    model = MLP(
        in_dim=input_dim,
        out_dim=output_dim,
        hidden=config["hidden"],
        activation=config["activation"],
        dropout=config["dropout"],
    )

    # 5. Configurar otimizador
    optimizer = build_optimizer(
        model,
        optimizer_name=config["optimizer"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # 6. Contar parâmetros treináveis
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # 7. Treinar e medir tempo
    start_time = time.perf_counter()

    training_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=config["epochs"],
        patience=config["patience"],
    )

    elapsed_time = time.perf_counter() - start_time

    # 8. Avaliar treino e validação com os melhores pesos
    train_metrics = evaluate_model(
        model,
        train_loader,
        task,
        target_scaler,
    )

    val_metrics = evaluate_model(
        model,
        val_loader,
        task,
        target_scaler,
    )

    test_metrics = evaluate_model(
        model,
        test_loader,
        task,
        target_scaler,
    )

    # 9. Calcular gap de generalização
    if task == "regression":
        generalization_gap = (
            val_metrics["rmse"] - train_metrics["rmse"]
        )
        main_metric = "rmse"

    else:
        generalization_gap = (
            train_metrics["f1_macro"] - val_metrics["f1_macro"]
        )
        main_metric = "f1_macro"

    # 10. Retornar resultados
    return {
        "task": task,
        "seed": seed,
        "config": config.copy(),
        "model": model,
        "trainable_params": trainable_params,
        "history": training_result["history"],
        "best_epoch": training_result["best_epoch"],
        "best_val_loss": training_result["best_val_loss"],
        "epochs_trained": training_result["epochs_trained"],
        "elapsed_time": elapsed_time,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "generalization_gap": generalization_gap,
        "main_metric": main_metric,
        "loss_name": loss_name,
        "test_metrics": test_metrics,
    }

def run_multiple_seeds(
    datasets,
    task,
    config,
    seeds,
    target_scaler=None,
):
    results = []

    for seed in seeds:
        print(f"\nExecutando {task} | Seed: {seed}")

        result = run_experiment(
            datasets=datasets,
            task=task,
            config=config,
            seed=seed,
            target_scaler=target_scaler,
        )

        results.append(result)

        print(
            f"Métrica treino: {result['train_metrics'][result['main_metric']]:.4f}"
        )
        print(
            f"Métrica validação: {result['val_metrics'][result['main_metric']]:.4f}"
        )
        print(f"Tempo: {result['elapsed_time']:.2f}s")

    # Consolidar resultados
    rows = []

    for result in results:
        metric = result["main_metric"]

        rows.append({
            "seed": result["seed"],
            "train_metric": result["train_metrics"][metric],
            "val_metric": result["val_metrics"][metric],
            "generalization_gap": result["generalization_gap"],
            "best_epoch": result["best_epoch"],
            "epochs_trained": result["epochs_trained"],
            "elapsed_time": result["elapsed_time"],
            "trainable_params": result["trainable_params"],
        })

    results_df = pd.DataFrame(rows)

    summary = results_df.drop(columns="seed").agg(["mean", "std"]).T

    return results, results_df, summary

def run_hyperparameter_experiment(
    datasets,
    task,
    base_config,
    parameter,
    values,
    seeds,
    target_scaler=None,
):
    all_results = []
    all_rows = []

    for value in values:
        # Copiar a baseline para não modificá-la
        config = copy.deepcopy(base_config)

        if parameter == "neurons":
            config["hidden"] = (value, value)

        elif parameter == "layers":
            config["hidden"] = tuple([config["hidden"][0]] * value)

        elif parameter == "loss":
            config["loss"] = value

        elif parameter in config:
            config[parameter] = value

        else:
            raise ValueError(f"Parâmetro desconhecido: {parameter}")

        print(f"\n{'=' * 50}")
        print(f"Tarefa: {task}")
        print(f"Hiperparâmetro: {parameter} = {value}")
        print(f"Arquitetura: {config['hidden']}")
        print(f"{'=' * 50}")

        # Executar com as cinco sementes
        results, results_df, summary = run_multiple_seeds(
            datasets=datasets,
            task=task,
            config=config,
            seeds=seeds,
            target_scaler=target_scaler,
        )

        all_results.append({
            "value": value,
            "config": config,
            "results": results,
            "summary": summary,
        })

        results_df = results_df.copy()
        results_df["tested_value"] = value

        all_rows.append(results_df)

    # Consolidar resultados individuais
    detailed_df = pd.concat(all_rows, ignore_index=True)

    # Calcular média e desvio-padrão por configuração
    metric_columns = [
        "train_metric",
        "val_metric",
        "generalization_gap",
        "best_epoch",
        "epochs_trained",
        "elapsed_time",
        "trainable_params",
    ]

    summary_df = (
        detailed_df
        .groupby("tested_value")[metric_columns]
        .agg(["mean", "std"])
    )

    return all_results, detailed_df, summary_df

def build_loss_function(loss_name, task, train_dataset):
    """
    Cria a função de perda conforme a tarefa.
    Os pesos da classificação são calculados somente
    com os rótulos do conjunto de treinamento.
    """

    name = loss_name.lower()

    if task == "regression":

        if name == "mse":
            return nn.MSELoss()

        elif name == "mae":
            return nn.L1Loss()

        elif name == "huber":
            return nn.HuberLoss(delta=1.0)

        else:
            raise ValueError(f"Loss de regressão inválida: {loss_name}")

    elif task == "classification":

        if name == "cross_entropy":
            return nn.CrossEntropyLoss()

        elif name == "cross_entropy_weighted":

            y_train = train_dataset.tensors[1].long()

            num_classes = 3

            class_counts = torch.bincount(
                y_train,
                minlength=num_classes
            )

            if torch.any(class_counts == 0):
                raise ValueError(
                    "Uma ou mais classes não aparecem no treinamento."
                )

            total_samples = len(y_train)

            class_weights = total_samples / (
                num_classes * class_counts.float()
            )

            print("Frequência das classes:", class_counts.tolist())
            print("Pesos das classes:", class_weights.tolist())

            return nn.CrossEntropyLoss(weight=class_weights)

        else:
            raise ValueError(
                f"Loss de classificação inválida: {loss_name}"
            )

    else:
        raise ValueError("Tarefa inválida.")

def run_interaction_experiment(
    datasets,
    task,
    base_config,
    neurons_values,
    weight_decay_values,
    seeds,
    target_scaler=None,
):
    all_results = []
    all_rows = []

    for neurons in neurons_values:
        for weight_decay in weight_decay_values:

            # Copiar a configuração de referência
            config = copy.deepcopy(base_config)

            # Modificar os dois hiperparâmetros
            config["hidden"] = (neurons, neurons)
            config["weight_decay"] = weight_decay

            print(f"\n{'=' * 50}")
            print(f"Tarefa: {task}")
            print(f"Neurônios: {neurons}")
            print(f"Weight decay: {weight_decay}")
            print(f"Arquitetura: {config['hidden']}")
            print(f"{'=' * 50}")

            # Executar com as mesmas sementes dos experimentos anteriores
            results, results_df, summary = run_multiple_seeds(
                datasets=datasets,
                task=task,
                config=config,
                seeds=seeds,
                target_scaler=target_scaler,
            )

            # Guardar resultados
            all_results.append({
                "neurons": neurons,
                "weight_decay": weight_decay,
                "config": config,
                "results": results,
                "summary": summary,
            })

            results_df = results_df.copy()
            results_df["neurons"] = neurons
            results_df["weight_decay"] = weight_decay

            all_rows.append(results_df)

    # Consolidar resultados individuais
    detailed_df = pd.concat(all_rows, ignore_index=True)

    # Resumir as métricas de cada combinação
    metric_columns = [
        "train_metric",
        "val_metric",
        "generalization_gap",
        "best_epoch",
        "epochs_trained",
        "elapsed_time",
        "trainable_params",
    ]

    summary_df = (
        detailed_df
        .groupby(["neurons", "weight_decay"])[metric_columns]
        .agg(["mean", "std"])
    )

    return all_results, detailed_df, summary_df

def run_random_search(
    datasets,
    task,
    base_config,
    search_space,
    seeds,
    n_configs=20,
    search_seed=42,
    target_scaler=None,
):
    all_results = []
    all_rows = []

    # 1. Gerar todas as combinações possíveis
    keys = list(search_space.keys())

    combinations = list(
        product(*(search_space[key] for key in keys))
    )

    if n_configs > len(combinations):
        raise ValueError(
            "Número de configurações maior que as combinações disponíveis."
        )

    # 2. Sortear configurações sem repetição
    rng = np.random.default_rng(search_seed)

    selected_indices = rng.choice(
        len(combinations),
        size=n_configs,
        replace=False,
    )

    selected_configs = [
        dict(zip(keys, combinations[index]))
        for index in selected_indices
    ]

    # 3. Executar cada configuração
    for i, sampled_config in enumerate(selected_configs, start=1):

        config = copy.deepcopy(base_config)

        # Definir arquitetura
        config["hidden"] = tuple(
            [sampled_config["neurons"]] * sampled_config["layers"]
        )

        # Definir demais hiperparâmetros
        for parameter, value in sampled_config.items():

            if parameter not in ["layers", "neurons"]:
                config[parameter] = value

        print(f"\n{'=' * 60}")
        print(f"Tarefa: {task}")
        print(f"Configuração: {i}/{n_configs}")
        print(f"Arquitetura: {config['hidden']}")
        print(f"Hiperparâmetros: {config}")
        print(f"{'=' * 60}")

        # Executar com as mesmas seeds dos experimentos anteriores
        results, results_df, summary = run_multiple_seeds(
            datasets=datasets,
            task=task,
            config=config,
            seeds=seeds,
            target_scaler=target_scaler,
        )

        # Guardar resultados
        all_results.append({
            "config_id": i,
            "config": config,
            "results": results,
            "summary": summary,
        })

        results_df = results_df.copy()
        results_df["config_id"] = i

        for parameter, value in sampled_config.items():
            results_df[parameter] = value

        all_rows.append(results_df)

    # 4. Consolidar resultados individuais
    detailed_df = pd.concat(
        all_rows,
        ignore_index=True,
    )

    # 5. Calcular média e desvio-padrão
    metric_columns = [
        "train_metric",
        "val_metric",
        "generalization_gap",
        "best_epoch",
        "epochs_trained",
        "elapsed_time",
        "trainable_params",
    ]

    summary_df = (
        detailed_df
        .groupby("config_id")[metric_columns]
        .agg(["mean", "std"])
    )

    # 6. Criar ranking pela métrica de validação
    ascending = task == "regression"

    ranking_df = (
        detailed_df
        .groupby("config_id")["val_metric"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=ascending)
    )

    # 7. Recuperar a melhor configuração
    best_config_id = ranking_df.index[0]

    best_result = next(
        result for result in all_results
        if result["config_id"] == best_config_id
    )

    print("\n" + "=" * 60)
    print(f"MELHOR CONFIGURAÇÃO — {task.upper()}")
    print("=" * 60)

    print("Hiperparâmetros:")
    print(best_result["config"])

    print("\nResultado na validação:")
    print(ranking_df.loc[best_config_id])

    return (
        all_results,
        detailed_df,
        summary_df,
        ranking_df,
        best_result,
    )