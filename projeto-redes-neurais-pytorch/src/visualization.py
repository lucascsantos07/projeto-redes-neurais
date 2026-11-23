"""
Módulo de Visualização Gráfica
Gera gráficos acadêmicos e analíticos padronizados para o relatório e notebooks:
- Curvas de aprendizado (treino vs validação com marcação do early stopping)
- Análise de resíduos e dispersão Real vs Previsto (Regressão)
- Curvas ROC e Precision-Recall (Classificação Multiclasse)
- Análise de variação de Limiar de Decisão (Threshold)
- Matrizes de confusão
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_learning_curves(history, best_epoch, task_title, metric_label, save_path=None):
    """
    Plota as curvas de perda de treino e validação destacando a melhor época.
    """
    plt.figure(figsize=(9, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.plot(epochs, history["train_loss"], label="Treino", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validação", color="#ff7f0e", linewidth=2)

    if best_epoch is not None:
        plt.axvline(
            x=best_epoch,
            color="#d62728",
            linestyle="--",
            label=f"Melhor época ({best_epoch})",
            linewidth=1.5,
        )

    plt.title(f"Curvas de Aprendizado - {task_title}", fontsize=13, fontweight="bold")
    plt.xlabel("Época", fontsize=11)
    plt.ylabel(metric_label, fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_regression_residuals(y_true, y_pred, save_path=None):
    """
    Plota dois gráficos para regressão na escala original:
    1. Real vs Previsto com linha de identidade
    2. Resíduos (y_true - y_pred) vs Previsto com linha zero
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Real vs Previsto
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05

    axes[0].scatter(y_true, y_pred, alpha=0.4, color="#2ca02c", edgecolors="none")
    axes[0].plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Identidade (Ideal)",
    )
    axes[0].set_title("Valores Reais vs. Previstos", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Valor Real (y)", fontsize=11)
    axes[0].set_ylabel("Valor Previsto (ŷ)", fontsize=11)
    axes[0].set_xlim(min_val - margin, max_val + margin)
    axes[0].set_ylim(min_val - margin, max_val + margin)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Gráfico 2: Resíduos vs Previsto
    axes[1].scatter(y_pred, residuals, alpha=0.4, color="#1f77b4", edgecolors="none")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Análise de Resíduos (e = y - ŷ)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Valor Previsto (ŷ)", fontsize=11)
    axes[1].set_ylabel("Resíduo (e)", fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_and_pr_curves(roc_pr_data, save_path=None):
    """
    Plota as curvas ROC e Precision-Recall lado a lado para classificação multiclasse.
    """
    fpr = roc_pr_data["fpr"]
    tpr = roc_pr_data["tpr"]
    roc_auc = roc_pr_data["roc_auc"]
    precision = roc_pr_data["precision"]
    recall = roc_pr_data["recall"]
    pr_auc = roc_pr_data["pr_auc"]
    class_names = roc_pr_data["class_names"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Curva ROC
    for i, name in enumerate(class_names):
        axes[0].plot(
            fpr[i], tpr[i],
            label=f"{name} (AUC = {roc_auc[i]:.3f})",
            linewidth=1.8
        )
    axes[0].plot(
        fpr["macro"], tpr["macro"],
        label=f"Macro-Média (AUC = {roc_auc['macro']:.3f})",
        color="black", linestyle=":", linewidth=2.2
    )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aleatório (AUC = 0.50)")
    axes[0].set_title("Curvas ROC (One-vs-Rest)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=11)
    axes[0].set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.3)

    # Curva Precision-Recall
    for i, name in enumerate(class_names):
        axes[1].plot(
            recall[i], precision[i],
            label=f"{name} (PR-AUC = {pr_auc[i]:.3f})",
            linewidth=1.8
        )
    axes[1].plot(
        recall["macro"], precision["macro"],
        label=f"Macro-Média (PR-AUC = {pr_auc['macro']:.3f})",
        color="black", linestyle=":", linewidth=2.2
    )
    axes[1].set_title("Curvas Precision-Recall (One-vs-Rest)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Revocação (Recall)", fontsize=11)
    axes[1].set_ylabel("Precisão (Precision)", fontsize=11)
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_analysis(y_true, y_probs, class_idx=1, class_name=None, save_path=None):
    """
    Analisa e plota o efeito da variação do limiar de decisão sobre Precision e Recall
    para a classe de interesse (definida na validação).
    """
    y_bin = (np.asarray(y_true) == class_idx).astype(int)
    probs = y_probs[:, class_idx]

    thresholds = np.linspace(0.05, 0.95, 50)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        pred_bin = (probs >= t).astype(int)
        tp = np.sum((pred_bin == 1) & (y_bin == 1))
        fp = np.sum((pred_bin == 1) & (y_bin == 0))
        fn = np.sum((pred_bin == 0) & (y_bin == 1))

        p = tp / max(tp + fp, 1e-8)
        r = tp / max(tp + fn, 1e-8)
        f = 2 * p * r / max(p + r, 1e-8)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    plt.figure(figsize=(8, 4.8))
    title_suffix = f" - {class_name}" if class_name else ""
    plt.plot(thresholds, precisions, label="Precisão", color="#1f77b4", linewidth=2)
    plt.plot(thresholds, recalls, label="Revocação (Recall)", color="#2ca02c", linewidth=2)
    plt.plot(thresholds, f1s, label="F1-Score", color="#d62728", linestyle="--", linewidth=2)

    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    plt.axvline(best_t, color="black", linestyle=":", label=f"Melhor Limiar F1 ({best_t:.2f})")

    plt.title(f"Trade-off Precision vs. Recall pelo Limiar{title_suffix}", fontsize=12, fontweight="bold")
    plt.xlabel("Limiar de Decisão (Threshold)", fontsize=11)
    plt.ylabel("Pontuação", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
