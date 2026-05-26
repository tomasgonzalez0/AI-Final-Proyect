# FoodVision AI - Development Plan for OpenCode / Codex 5.2

## Project Overview

**Project:** FoodVision AI - Intelligent Restaurant Management System  
**Tech Stack:** Python 3.10+, scikit-learn, TensorFlow/Keras, pandas, matplotlib, seaborn  
**Entry Point:** `src/main.py`  
**Dataset:** `data/restaurante_foodvision.xlsx`

---

## Folder Structure

```
foodvision_ai/
|-- data/
|   |-- restaurante_foodvision.xlsx          # Raw dataset (user provides)
|   |-- cleaned_dataset.csv                  # Auto-generated after cleaning
|
|-- src/
|   |-- main.py                              # Application entry point (menu-driven CLI)
|   |-- data/
|   |   |-- __init__.py
|   |   |-- loader.py                        # Excel reading and raw display
|   |   |-- cleaner.py                       # Automated data cleaning
|   |   |-- transformer.py                   # Encoding, scaling, feature engineering
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   |-- ml_models.py                     # Decision Tree, Random Forest, Logistic Reg, KNN
|   |   |-- deep_learning.py                 # Shallow NN and Deep NN (Keras)
|   |   |-- evaluator.py                     # Metrics for classification and regression
|   |   |-- comparator.py                    # Side-by-side model comparison table
|   |
|   |-- visualization/
|   |   |-- __init__.py
|   |   |-- plots.py                         # All matplotlib/seaborn charts
|   |
|   |-- utils/
|       |-- __init__.py
|       |-- logger.py                        # Simple console logger (no emojis)
|       |-- config.py                        # Paths, constants, random seeds
|
|-- outputs/
|   |-- plots/                               # All saved PNG charts
|   |-- reports/                             # CSV or TXT metric summaries
|   |-- models/                              # Saved .pkl and .h5 model files
|
|-- tests/
|   |-- test_cleaner.py
|   |-- test_transformer.py
|   |-- test_models.py
|
|-- docs/
|   |-- DEVELOPMENT_PLAN.md                  # This file
|   |-- OPENCODE_PROMPT.md                   # Prompt for OpenCode session
|
|-- requirements.txt
|-- README.md
```

---

## Module Implementation Order

Implement modules strictly in this order. Each module must be complete and tested before moving to the next.

### Phase 1 - Data Layer

**Module 1: `utils/config.py`**
- Define all constants: file paths, random seed (42), test split ratio (0.2)
- Define target variable names per task
- No external dependencies

**Module 2: `utils/logger.py`**
- Simple logging wrapper using Python `logging` module
- Format: `[LEVEL] YYYY-MM-DD HH:MM:SS - message`
- No emojis, no special characters

**Module 3: `data/loader.py`**
- Function `load_excel(filepath) -> pd.DataFrame`
- Function `display_dataset_info(df)`: shape, dtypes, head(10), describe()
- Raise `FileNotFoundError` with clear message if file is missing

**Module 4: `data/cleaner.py`**
- Function `clean_dataset(df) -> pd.DataFrame` that runs the full pipeline:
  - `standardize_city_names(df)`: lowercase then title-case column `ciudad`
  - `fix_negative_values(df)`: set negative `tiempo_preparacion` and `valor_total` to NaN
  - `fix_outliers(df)`: cap `tiempo_preparacion` at 120 minutes max
  - `impute_missing_values(df)`: median for numeric, mode for categorical
  - `remove_duplicates(df)`: drop exact duplicate rows
  - `validate_ranges(df)`: assert `calificacion_cliente` in [1,5]
- Each sub-function logs a summary of changes made (rows affected, values changed)
- Return the cleaned DataFrame and a dict `cleaning_report` with counts

**Module 5: `data/transformer.py`**
- Function `apply_minmax_scaling(df, columns) -> df`
- Function `apply_zscore_scaling(df, columns) -> df`
- Function `apply_one_hot_encoding(df, columns) -> df`
- Function `apply_label_encoding(df, columns) -> df, encoders_dict`
- Function `create_features(df) -> df`:
  - `categoria_consumo`: bins on `valor_total` -> "Bajo", "Medio", "Alto"
  - `nivel_demora`: derive from `tiempo_preparacion` -> "Normal", "Moderado", "Critico"
  - `cliente_premium`: `cliente_frecuente == "Si"` AND `calificacion_cliente >= 4`
  - `promedio_compra_cliente`: rolling mean per unique session (use `valor_total` / `cantidad_productos`)
- Function `apply_pca(df, n_components) -> df, pca_object`
- Function `prepare_for_modeling(df, target_col) -> X_train, X_test, y_train, y_test`

---

### Phase 2 - Models Layer

**Module 6: `models/ml_models.py`**
- Class `MLModelTrainer` with methods:
  - `train_decision_tree(X_train, y_train, task) -> model`
  - `train_random_forest(X_train, y_train, task) -> model`
  - `train_logistic_regression(X_train, y_train) -> model`
  - `train_knn(X_train, y_train, task) -> model`
  - `predict(model, X_test) -> y_pred`
- Parameter `task` accepts "classification" or "regression"
- All models use `random_state=42` where applicable
- Save each trained model to `outputs/models/` using `joblib`

**Module 7: `models/deep_learning.py`**
- Function `build_shallow_nn(input_dim, output_units, task) -> model`
  - Architecture: Input -> Dense(64, relu) -> Dense(output)
- Function `build_deep_nn(input_dim, output_units, task) -> model`
  - Architecture: Input -> Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.2) -> Dense(32, relu) -> Dense(output)
- Function `train_neural_network(model, X_train, y_train, epochs, batch_size) -> history`
  - Default: epochs=50, batch_size=32
  - Use `validation_split=0.2`
  - Record training time using `time.time()`
- Function `predict_nn(model, X_test) -> y_pred`
- Save model to `outputs/models/` as `.h5`

**Module 8: `models/evaluator.py`**
- Function `evaluate_classification(y_true, y_pred, model_name) -> dict`
  - Metrics: accuracy, precision (weighted), recall (weighted), f1 (weighted)
  - Print confusion matrix to console
- Function `evaluate_regression(y_true, y_pred, model_name) -> dict`
  - Metrics: MAE, MSE, RMSE, R2
- Function `print_metrics_table(results_list)`: formatted console table

**Module 9: `models/comparator.py`**
- Function `compare_all_models(results_list) -> pd.DataFrame`
  - Accepts list of dicts from evaluator functions
  - Returns DataFrame sorted by main metric (accuracy or R2)
- Function `save_comparison_report(df, filepath)`: save to CSV

---

### Phase 3 - Visualization Layer

**Module 10: `visualization/plots.py`**
- All functions save PNG to `outputs/plots/` AND show with `plt.show()`
- Function `plot_data_distribution(df)`: histograms for numeric columns
- Function `plot_correlation_heatmap(df)`: seaborn heatmap
- Function `plot_confusion_matrix(y_true, y_pred, model_name)`: annotated heatmap
- Function `plot_model_comparison(results_df)`: horizontal bar chart
- Function `plot_training_history(history, model_name)`: loss and accuracy curves
- Function `plot_feature_importance(model, feature_names)`: for tree-based models
- Function `plot_pca_variance(pca_object)`: explained variance ratio
- All charts must have: title, axis labels, legend where applicable, saved filename logged

---

### Phase 4 - Application Entry Point

**Module 11: `src/main.py`**
- Menu-driven CLI with numbered options:
  ```
  === FoodVision AI - Restaurant Intelligence System ===
  1. Load and display dataset
  2. Clean data automatically
  3. Transform variables
  4. Train ML models (classification)
  5. Train ML models (regression)
  6. Train Neural Networks
  7. Compare all models
  8. Generate visualizations
  9. Make a prediction (interactive)
  0. Exit
  ```
- Each option calls the corresponding module function
- State is held in module-level variables (df_raw, df_clean, df_transformed, results)
- Option 9 prompts user for feature values and returns prediction from best model
- Wrap all operations in try/except with clear error messages

---

## Coding Standards (Mandatory)

1. **No emojis or special characters** in any string, comment, log, or print statement
2. **Docstrings on every function** using Google style:
   ```python
   def function_name(param):
       """
       Brief description.

       Args:
           param (type): Description.

       Returns:
           type: Description.
       """
   ```
3. **Type hints on all function signatures**
4. **Constants in config.py** - no magic numbers inline
5. **Imports grouped**: stdlib, then third-party, then local - separated by blank line
6. **Max line length: 100 characters**
7. **No bare except clauses** - always specify exception type
8. **random_state=42** on all stochastic operations

---

## Target Variables Per Task

| Task | Target Column | Type |
|------|--------------|------|
| Predict delivery delay | `demora_entrega` | Classification (binary) |
| Predict customer satisfaction | `satisfaccion_cliente` | Classification (multi-class) |
| Predict daily sales | `ventas_dia` | Regression |

---

## Dataset Column Reference

| Column | Type | Notes |
|--------|------|-------|
| id_pedido | int | Identifier, drop before modeling |
| ciudad | str | Categorical - standardize case |
| tipo_comida | str | Categorical - encode |
| hora_pedido | int/str | May need extraction (hour of day) |
| cantidad_productos | int | Numeric |
| valor_total | float | Numeric - check negatives |
| tiempo_preparacion | int | Numeric - check negatives and outliers |
| metodo_pago | str | Categorical - encode |
| cliente_frecuente | str | Binary Yes/No |
| calificacion_cliente | int | 1-5 range validation |
| clima | str | Categorical - encode |
| demora_entrega | str | Target for classification |
| satisfaccion_cliente | str | Target for classification |
| ventas_dia | float | Target for regression |

---

## Error Handling Requirements

- File not found: print path and suggest placing file in `data/` folder
- Empty DataFrame after cleaning: raise `ValueError` with row count before/after
- Model training failure: catch, log error, continue to next model
- Plot failure: log warning, skip plot, do not crash application
- Invalid user input in menu: print "Invalid option, please try again"

---

## Dependencies

Core (recommended for most of the project):

- `requirements.txt`

Optional (only needed for `src/models/deep_learning.py`):

- `requirements-deep-learning.txt`
