from pathlib import Path

# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Diretório dos datasets
DATA_DIR = PROJECT_ROOT / "data" / "processed"

REGRESSION_DATA_PATH = DATA_DIR / "dataset_regression.csv"
CLASSIFICATION_DATA_PATH = DATA_DIR / "dataset_classification.csv"

# Configurações da divisão dos dados
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # Proporção retirada do conjunto de treino

# Alvos
REGRESSION_TARGET = "Diabetes_Risk_Score"
CLASSIFICATION_TARGET = "Diabetes_Risk"

# Features numéricas compartilhadas
NUMERIC_FEATURES = [
    "Fasting_Blood_Sugar",
    "HbA1c",
    "Age",
    "BMI",
]

# Features categóricas de regressão
REGRESSION_CATEGORICAL_FEATURES = [
    "Family_History_Diabetes",
    "Hypertension",
    "Diet_Quality",
    "Physical_Activity_Level",
]

# Features categóricas de classificação
CLASSIFICATION_CATEGORICAL_FEATURES = [
    "Family_History_Diabetes",
    "Hypertension",
    "Diet_Quality",
    "PCOS",
    "Heart_Disease",
]

# Ordem explícita das classes da classificação
CLASS_NAMES = ["Low", "Moderate", "High"]
CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}
