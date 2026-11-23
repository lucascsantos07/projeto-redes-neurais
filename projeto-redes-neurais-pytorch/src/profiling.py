"""
Módulo de Eficiência Computacional (Profiling)
Implementa medições rigorosas de tempo, memória, throughput, complexidade teórica (MACs),
tamanho de arquivos e pegada de carbono conforme Aula 08 e Diretrizes da Atividade.
"""

import os
import time
import psutil
import numpy as np
import torch


def measure_inference_efficiency(model, X_tensor, n_rep=30, n_warmup=5, device="cpu"):
    """
    Mede tempo, latência e memória residente da etapa de inferência
    conforme Listing 4 de main_10.pdf e Aula 08.

    Args:
        model: Rede neural PyTorch.
        X_tensor: Tensor com os dados de entrada.
        n_rep: Número de repetições cronometradas (padrão 30).
        n_warmup: Número de passagens de aquecimento (padrão 5).
        device: Dispositivo de execução ('cpu' ou 'cuda').

    Returns:
        Dicionário com estatísticas consolidadas.
    """
    model.to(device)
    model.eval()
    X_dev = X_tensor.to(device)

    proc = psutil.Process()

    # Passagens de aquecimento (Warm-up)
    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(X_dev)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

    # Medição de memória antes da repetição
    mem_antes = proc.memory_info().rss

    tempos = []
    with torch.inference_mode():
        for _ in range(n_rep):
            inicio = time.perf_counter()
            _ = model(X_dev)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            tempos.append(time.perf_counter() - inicio)

    mem_depois = proc.memory_info().rss

    media_s = float(np.mean(tempos))
    desvio_s = float(np.std(tempos))
    n_samples = len(X_tensor)

    return {
        "tempo_medio_total_s": media_s,
        "tempo_desvio_s": desvio_s,
        "tempo_por_amostra_ms": 1000.0 * media_s / max(n_samples, 1),
        "vazao_amostras_s": n_samples / max(media_s, 1e-9),
        "delta_memoria_kb": (mem_depois - mem_antes) / 1024.0,
        "memoria_rss_mb": mem_depois / (1024.0 * 1024.0),
        "num_threads_cpu": torch.get_num_threads(),
        "dispositivo": str(device),
    }


def compare_batch_vs_single_inference(model, X_tensor, device="cpu", n_rep=30):
    """
    Compara o tempo de inferência em lote versus amostra individual (one-by-one).
    """
    model.to(device)
    model.eval()

    # 1. Inferência em lote
    res_lote = measure_inference_efficiency(model, X_tensor, n_rep=n_rep, device=device)

    # 2. Inferência de uma amostra única (primeira amostra)
    X_single = X_tensor[:1]
    res_single = measure_inference_efficiency(model, X_single, n_rep=n_rep, device=device)

    # 3. Tempo sequencial total estimado se rodar amostra por amostra
    tempo_sequencial_estimado_s = res_single["tempo_medio_total_s"] * len(X_tensor)
    speedup_lote = tempo_sequencial_estimado_s / max(res_lote["tempo_medio_total_s"], 1e-9)

    return {
        "lote_tempo_total_s": res_lote["tempo_medio_total_s"],
        "lote_tempo_por_amostra_ms": res_lote["tempo_por_amostra_ms"],
        "lote_vazao_amostras_s": res_lote["vazao_amostras_s"],
        "amostra_tempo_por_amostra_ms": res_single["tempo_por_amostra_ms"],
        "amostra_vazao_amostras_s": res_single["vazao_amostras_s"],
        "razao_aceleracao_lote": speedup_lote,
    }


def compute_model_complexity(model, input_dim):
    """
    Calcula número de parâmetros treináveis, camadas lineares e
    operações teóricas Multiply-Accumulate (MACs) e FLOPs por amostra.
    """
    total_params = 0
    trainable_params = 0
    num_linear_layers = 0
    total_macs = 0

    curr_in = input_dim
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            num_linear_layers += 1
            in_f = module.in_features
            out_f = module.out_features
            # Cada neurônio executa in_f multiplicações e adições
            total_macs += in_f * out_f

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    # Cada MAC corresponde aproximadamente a 2 FLOPs (multiplicação + adição)
    flops = total_macs * 2
    param_size_kb = (trainable_params * 4) / 1024.0  # float32 = 4 bytes

    return {
        "parametros_treinaveis": trainable_params,
        "parametros_totais": total_params,
        "camadas_lineares": num_linear_layers,
        "macs_por_amostra": total_macs,
        "flops_por_amostra": flops,
        "tamanho_teorico_pesos_kb": param_size_kb,
    }


def get_checkpoint_disk_size(checkpoint_path):
    """
    Retorna o tamanho do arquivo .pt salvo em disco em KB.
    """
    if os.path.exists(checkpoint_path):
        return os.path.getsize(checkpoint_path) / 1024.0
    return 0.0


def estimate_energy_and_carbon(time_seconds, power_watts=65.0, emission_factor_brazil=45.8):
    """
    Estima a energia consumida (kWh) e as emissões de CO2 (gCO2)
    usando o fator oficial do Brasil (MCTI/SIRENE: 45,8 gCO2/kWh).

    Args:
        time_seconds: Duração em segundos.
        power_watts: Potência média estimada do processador/GPU (padrão 65W TDP médio).
        emission_factor_brazil: Fator de emissão nacional em gCO2/kWh (45,8 conforme aula 08).
    """
    hours = time_seconds / 3600.0
    energy_kwh = (power_watts * hours) / 1000.0
    carbon_gco2 = energy_kwh * emission_factor_brazil

    return {
        "tempo_segundos": time_seconds,
        "potencia_watts": power_watts,
        "energia_kwh": energy_kwh,
        "emissao_co2_g": carbon_gco2,
        "fator_emissao_utilizado": emission_factor_brazil,
    }
