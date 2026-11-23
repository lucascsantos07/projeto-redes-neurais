# Redes Neurais com PyTorch: Classificação e Regressão

**Disciplina:** DCT1401 - Inteligência Artificial (BSI - UFRN)  
**Professor:** Thommas K. S. Flores  
**Atividade:** Redes Neurais Artificiais com PyTorch: Classificação, Regressão (Atividade Prática Avaliativa)

---

## 📌 Descrição do Projeto

Este projeto desenvolve e analisa redes neurais artificiais do tipo Perceptron Multicamadas (MLP) em **PyTorch** para tarefas de **Regressão** e **Classificação**. O objetivo é verificar a viabilidade das redes neurais sobre dados tabulares previamente estudados, analisar o comportamento do modelo sob diferentes decisões de projeto (hiperparâmetros), garantir reprodutibilidade experimental e persistir os modelos para inferência fora do ambiente de treinamento.

---

## 🗂️ Estrutura do Repositório

```text
projeto-redes-neurais-pytorch/
├── README.md                      # Descrição do projeto e instruções de uso
├── requirements.txt               # Dependências do ambiente Python
├── main_10.pdf                    # Guia de diretrizes da atividade prática
│
├── data/
│   └── processed/                 # Datasets processados, tensores e scalers
│       ├── dataset_regression.csv
│       ├── dataset_classification.csv
│       ├── regression_data.pt
│       ├── classification_data.pt
│       ├── regression_preprocessor.pkl
│       ├── classification_preprocessor.pkl
│       ├── regression_target_scaler.pkl
│       ├── regression_feature_names.pt
│       ├── classification_feature_names.pt
│       ├── regression_samples.pt
│       └── classification_samples.pt
│
├── models/                        # Checkpoints dos modelos treinados (.pt)
│   ├── modelo_regressao.pt
│   ├── modelo_classificacao.pt
│   ├── regression_original_outputs.pt
│   └── classification_original_outputs.pt
│
├── notebooks/                     # Notebooks Jupyter numerados e organizados
│   ├── 01_preparacao_dados.ipynb              # Etapa 1: Protocolo experimental e tensores
│   ├── 03_mlp_regressao.ipynb                 # Etapa 2: MLP base para regressão
│   ├── 04_mlp_classificacao.ipynb             # Etapa 2: MLP base para classificação
│   ├── 05_hiperparametros.ipynb               # Etapa 3: Estudo sistemático e busca final
│   ├── 06_salvar_carregar_inferencia.ipynb    # Etapa 4: Checkpoints, fidelidade e inferência
│   ├── 07_avaliacao_generalizacao.ipynb       # Etapa 5: (Pendente) Métricas e generalização
│   └── 08_eficiencia_computacional.ipynb      # Etapa 5/6: (Pendente) Custo computacional
│
├── reports/                       # Artefatos gerados nos experimentos
│   ├── figures/                   # Curvas de perda, mapas de calor e sensibilidade
│   └── tables/                    # Tabelas consolidadas em formato LaTeX (.tex)
│
└── src/                           # Módulos Python reutilizáveis
    ├── config.py                  # Constantes, caminhos e sementes do projeto
    ├── experiments.py             # Funções de execução de experimentos e grid search
    ├── inference.py               # Pipeline autônomo de carregamento e inferência
    ├── models.py                  # Classe MLP parametrizável em PyTorch
    ├── train.py                   # Laço de treinamento com validação e early stopping
    ├── metrics.py                 # Funções de métricas (Etapa 5)
    ├── profiling.py               # Medição de tempo, memória e energia (Etapa 5)
    ├── visualization.py           # Funções auxiliares para gráficos (Etapa 5)
    ├── data/                      # Módulos de carregamento e pré-processamento
    │   ├── load_data.py
    │   ├── preprocessing.py
    │   ├── split_data.py
    │   └── torch_data.py
    └── utils/
        └── reproducibility.py     # Controle de sementes aleatórias (random, numpy, torch)
```

---

## 🚦 Status de Desenvolvimento da Atividade

| Etapa | Descrição | Status | Detalhes |
|---|---|:---:|---|
| **1. Preparação dos Dados** | Protocolo experimental, divisões estratificadas, StandardScaler e tensores PyTorch | Concluído | `01_preparacao_dados.ipynb` |
| **2. Modelo Base (MLP)** | Construção do MLP parametrizável, loss functions e laço com early stopping | Concluído | `03_mlp_regressao.ipynb`<br>`04_mlp_classificacao.ipynb` |
| **3. Estudo de Hiperparâmetros** | Avaliação sistemática (neurônios, camadas, ativação, otimizador, lr, batch, weight decay, perda, interação e busca em grade) | Concluído | `05_hiperparametros.ipynb` |
| **4. Persistência e Inferência** | Salvamento de checkpoint com pré-processamento, carregamento limpo e teste de fidelidade (`torch.allclose`) | Concluído | `06_salvar_carregar_inferencia.ipynb` |
| **5. Avaliação com Métricas** | Avaliação no teste, curvas ROC/PR, resíduos, gap e eficiência computacional | *Pendente* | `07_avaliacao_generalizacao.ipynb`<br>`08_eficiencia_computacional.ipynb` |
| **6. Análise e Conclusões** | Resumo final, sensibilidade, respostas às diretrizes e relatório técnico | *Pendente* | Relatório final em PDF |

---

## ⚙️ Instruções de Execução

### 1. Preparar o Ambiente
Recomenda-se criar um ambiente virtual dedicado:
```bash
python -m venv .venv
# Ativação no Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Instalação das dependências:
pip install -r requirements.txt
```

### 2. Ordem de Execução dos Notebooks
Para reproduzir os resultados de ponta a ponta:
1. `01_preparacao_dados.ipynb`: gera as matrizes divididas, scalers e tensores `.pt` em `data/processed/`.
2. `03_mlp_regressao.ipynb` e `04_mlp_classificacao.ipynb`: valida as arquiteturas baseline de regressão e classificação.
3. `05_hiperparametros.ipynb`: executa os experimentos multissemente, gera os gráficos/tabelas em `reports/` e treina os modelos finais salvos em `models/`.
4. `06_salvar_carregar_inferencia.ipynb`: valida o carregamento autônomo, verifica a fidelidade numérica e executa inferências pontuais e em lote.

---

## 🔮 Inferência com Modelos Salvos

Para carregar o modelo treinado de forma independente:
```python
import torch
from src.inference import load_model, predict_regression, predict_classification

# Carregar checkpoint completo (na CPU ou GPU)
checkpoint = torch.load("models/modelo_regressao.pt", map_location="cpu", weights_only=False)
model = load_model(checkpoint, device="cpu")

# Executar inferência sobre novos dados tabulares (DataFrame bruto)
predictions = predict_regression(model, checkpoint, novas_amostras_df)
```
