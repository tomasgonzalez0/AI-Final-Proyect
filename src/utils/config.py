"""Configuration constants for FoodVision AI."""

from __future__ import annotations

from pathlib import Path


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXCEL_FILENAME = "restaurante_foodvision.xlsx"

TARGET_CLASSIFICATION_BINARY = "demora_entrega"
TARGET_CLASSIFICATION_MULTI = "satisfaccion_cliente"
TARGET_REGRESSION = "ventas_dia"

COLUMNS_TO_DROP = ["id_pedido"]

CATEGORICAL_COLUMNS = [
    "ciudad",
    "tipo_comida",
    "metodo_pago",
    "cliente_frecuente",
    "clima",
]

NUMERIC_COLUMNS = [
    "hora_pedido",
    "cantidad_productos",
    "valor_total",
    "tiempo_preparacion",
    "calificacion_cliente",
]
