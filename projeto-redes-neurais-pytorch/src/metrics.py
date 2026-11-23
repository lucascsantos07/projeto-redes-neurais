"""
Módulo de Métricas de Avaliação - Conforme Aula 08 (Avaliação de Modelos de IA).
Implementa métricas completas para tarefas de Regressão e Classificação.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    median_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


# ==============================================================================
# 1. MÉTRICAS DE REGRESSÃO
# ==============================================================================

def compute_regression_metrics(y_true, y_pred):
    """
    Calcula as métricas obrigatórias da Aula 08 para tarefas de regressão.
    Importante: y_true e y_pred devem estar na escala original do alvo.

    Retorna um dicionário com:
    - MAE: Erro Médio Absoluto
    - MSE: Erro Quadrático Médio
    - RMSE: Raiz do Erro Quadrático Médio
    - MAPE: Erro Percentual Absoluto Médio (%)
    - R2: Coeficiente de Determinação
    - MedAE: Mediana do Erro Absoluto
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    mae = mean_absolute_error(y_t, y_p)
    mse = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)

    # Cálculo seguro do MAPE para evitar divisão por zero se houver valores nulos
    epsilon = 1e-8
    mape = np.mean(np.abs((y_t - y_p) / np.where(y_t == 0, epsilon, y_t))) * 100.0

    r2 = r2_score(y_t, y_p)
    medae = median_absolute_error(y_t, y_p)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "MedAE": medae,
    }


def compute_residuals(y_true, y_pred):
    """
    Calcula os resíduos: e_i = y_i - y_hat_i.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    return y_t - y_p


# ==============================================================================
# 2. MÉTRICAS DE CLASSIFICAÇÃO
# ==============================================================================

def compute_classification_metrics(y_true, y_pred, y_probs=None, class_names=None):
    """
    Calcula as métricas obrigatórias da Aula 08 para tarefas de classificação.
    Suporta cenários binários e multiclasse.
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_pred, dtype=np.int64).ravel()

    acc = accuracy_score(y_t, y_p)

    # Médias Macro e Weighted
    prec_macro = precision_score(y_t, y_p, average="macro", zero_division=0)
    prec_weighted = precision_score(y_t, y_p, average="weighted", zero_division=0)

    rec_macro = recall_score(y_t, y_p, average="macro", zero_division=0)
    rec_weighted = recall_score(y_t, y_p, average="weighted", zero_division=0)

    f1_macro = f1_score(y_t, y_p, average="macro", zero_division=0)
    f1_weighted = f1_score(y_t, y_p, average="weighted", zero_division=0)

    cm = confusion_matrix(y_t, y_p)

    metrics = {
        "Accuracy": acc,
        "Precision_macro": prec_macro,
        "Precision_weighted": prec_weighted,
        "Recall_macro": rec_macro,
        "Recall_weighted": rec_weighted,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "Confusion_Matrix": cm,
    }

    # Se probabilidades forem fornecidas, calcula ROC-AUC e PR-AUC
    if y_probs is not None:
        y_probs = np.asarray(y_probs, dtype=np.float64)
        unique_classes = np.unique(y_t)
        n_classes = len(unique_classes)

        if n_classes == 2 and y_probs.ndim == 1:
            metrics["ROC_AUC"] = roc_auc_score(y_t, y_probs)
            metrics["PR_AUC"] = average_precision_score(y_t, y_probs)
        elif n_classes == 2 and y_probs.ndim == 2:
            metrics["ROC_AUC"] = roc_auc_score(y_t, y_probs[:, 1])
            metrics["PR_AUC"] = average_precision_score(y_t, y_probs[:, 1])
        else:
            # Multiclasse One-vs-Rest (OvR)
            y_bin = label_binarize(y_t, classes=unique_classes)
            metrics["ROC_AUC_macro"] = roc_auc_score(
                y_bin, y_probs, multi_class="ovr", average="macro"
            )
            metrics["ROC_AUC_weighted"] = roc_auc_score(
                y_bin, y_probs, multi_class="ovr", average="weighted"
            )
            metrics["PR_AUC_macro"] = average_precision_score(
                y_bin, y_probs, average="macro"
            )
            metrics["PR_AUC_weighted"] = average_precision_score(
                y_bin, y_probs, average="weighted"
            )

    return metrics


# ==============================================================================
# 3. CURVAS ROC E PRECISION-RECALL (MULTICLASSE)
# ==============================================================================

def compute_multiclass_roc_pr(y_true, y_probs, class_names=None):
    """
    Calcula curvas ROC e Precision-Recall completas (per-class e macro)
    para classificação multiclasse (One-vs-Rest).
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_probs = np.asarray(y_probs, dtype=np.float64)
    classes = np.unique(y_t)
    n_classes = len(classes)

    if class_names is None:
        class_names = [f"Classe {c}" for c in classes]

    y_bin = label_binarize(y_t, classes=classes)

    fpr, tpr, roc_auc = {}, {}, {}
    precision, recall, pr_auc = {}, {}, {}

    for i in range(n_classes):
        # ROC
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # PR
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Macro-média ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    # Macro-média PR
    all_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        # Interpolação decrescente de recall
        mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
    mean_precision /= n_classes
    recall["macro"] = all_recall
    precision["macro"] = mean_precision
    pr_auc["macro"] = auc(all_recall, mean_precision)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "class_names": class_names,
    }
