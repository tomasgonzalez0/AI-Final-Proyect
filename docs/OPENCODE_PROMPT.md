# OpenCode Session Prompt - FoodVision AI

## Context

You are implementing a complete Python application called FoodVision AI for a restaurant chain
in Colombia. The application analyzes historical sales and operational data to predict delivery
delays, customer satisfaction, and daily sales using Machine Learning and Deep Learning.

The full development plan is in `docs/DEVELOPMENT_PLAN.md`. Read it completely before writing
any code.

## Your Immediate Task

Implement the entire application following the module order in the development plan exactly.
Do not skip phases. Do not implement modules out of order.

## Implementation Rules - Read Carefully

1. Read `docs/DEVELOPMENT_PLAN.md` fully before writing any code.
2. Implement one module at a time. Verify it runs without errors before moving to the next.
3. Every function must have a complete Google-style docstring with Args and Returns sections.
4. Every function must have type hints on parameters and return values.
5. No emojis, no special characters (arrows, bullets, stars) in any string, print, log, or comment.
6. All constants (file paths, seeds, thresholds) must live in `src/utils/config.py`.
7. Use `random_state=42` on every stochastic operation.
8. Wrap risky operations (file I/O, model training, plotting) in specific try/except blocks.
9. Log every significant action using the logger from `src/utils/logger.py`.
10. Test each module individually in the REPL before writing the next one.

## Module Implementation Sequence

Phase 1 - Data Layer (complete before Phase 2):
- src/utils/config.py
- src/utils/logger.py
- src/data/loader.py
- src/data/cleaner.py
- src/data/transformer.py

Phase 2 - Models Layer (complete before Phase 3):
- src/models/ml_models.py
- src/models/deep_learning.py
- src/models/evaluator.py
- src/models/comparator.py

Phase 3 - Visualization (complete before Phase 4):
- src/visualization/plots.py

Phase 4 - Entry Point (last):
- src/main.py

## Specific Technical Requirements Per Module

### src/utils/config.py
- Define: DATA_DIR, OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR as Path objects
- Define: RANDOM_STATE = 42, TEST_SIZE = 0.2, EXCEL_FILENAME
- Define: TARGET_CLASSIFICATION_BINARY = "demora_entrega"
- Define: TARGET_CLASSIFICATION_MULTI = "satisfaccion_cliente"
- Define: TARGET_REGRESSION = "ventas_dia"
- Define: COLUMNS_TO_DROP = ["id_pedido"]
- Define: CATEGORICAL_COLUMNS and NUMERIC_COLUMNS lists

### src/data/cleaner.py
- standardize_city_names: lowercase the column, then apply str.title()
- fix_negative_values: for columns tiempo_preparacion and valor_total, replace values < 0 with NaN
- fix_outliers: cap tiempo_preparacion at 120 using clip(lower=0, upper=120)
- impute_missing_values: SimpleImputer with median strategy for numeric, most_frequent for categorical
- remove_duplicates: df.drop_duplicates() and log how many rows were removed
- validate_ranges: assert calificacion_cliente between 1 and 5, set out-of-range to NaN then impute
- clean_dataset must return (cleaned_df, cleaning_report_dict)

### src/data/transformer.py
- apply_minmax_scaling: use MinMaxScaler, fit only on training columns, return scaler too
- apply_zscore_scaling: use StandardScaler, same pattern
- apply_one_hot_encoding: use pd.get_dummies with drop_first=True
- apply_label_encoding: use LabelEncoder per column, return dict of fitted encoders
- create_features:
  - categoria_consumo: pd.cut on valor_total into 3 equal-width bins labeled "Bajo", "Medio", "Alto"
  - nivel_demora: if tiempo_preparacion <= 20 then "Normal", <= 40 then "Moderado", else "Critico"
  - cliente_premium: boolean column, True where cliente_frecuente=="Si" and calificacion_cliente >= 4
  - promedio_compra_cliente: valor_total divided by cantidad_productos, fill inf with NaN then median
- apply_pca: use PCA from sklearn, return (transformed_df, fitted_pca)
- prepare_for_modeling: drop COLUMNS_TO_DROP, encode target if categorical, train_test_split with stratify where applicable

### src/models/ml_models.py
- For classification tasks: DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, KNeighborsClassifier
- For regression tasks: DecisionTreeRegressor, RandomForestRegressor, LinearRegression, KNeighborsRegressor
- The `task` parameter controls which variant is instantiated
- Save each model with joblib.dump to MODELS_DIR with filename pattern: {model_name}_{task}.pkl
- Return fitted model object from each train function

### src/models/deep_learning.py
- Use tensorflow.keras Sequential API
- Classification output: softmax activation, sparse_categorical_crossentropy loss
- Binary classification output: sigmoid activation, binary_crossentropy loss
- Regression output: linear activation, mse loss
- Record wall-clock training time using time.time() before and after model.fit()
- Return history object from train_neural_network
- Save model as {model_name}.h5 in MODELS_DIR

### src/models/evaluator.py
- evaluate_classification returns dict with keys: model_name, accuracy, precision, recall, f1_score, task
- evaluate_regression returns dict with keys: model_name, mae, mse, rmse, r2, task
- print_metrics_table uses tabulate or manual string formatting - NO external tabulate dependency required
- Always print results to console in addition to returning dict

### src/visualization/plots.py
- Figure size default: (12, 6) for most plots, (10, 8) for heatmaps
- DPI: 150 for saved files
- Color palette: use seaborn "Blues" or "viridis" consistently
- All save paths must use pathlib: PLOTS_DIR / f"{plot_name}.png"
- Call plt.tight_layout() before every savefig
- Close figures with plt.close() after saving to avoid memory leaks

### src/main.py
- Import all module functions at top of file
- Maintain state in a dict: app_state = {"df_raw": None, "df_clean": None, "df_transformed": None, "results": [], "best_model": None}
- Menu loop runs until user enters "0"
- Option 9 (interactive prediction):
  - Ask user to enter values for: ciudad, tipo_comida, hora_pedido, cantidad_productos, valor_total, tiempo_preparacion, metodo_pago, clima
  - Apply same preprocessing as training pipeline
  - Use best model from results to predict
  - Print prediction label and confidence if classification

## Expected Outputs When Running main.py

When the user runs through all menu options sequentially, the outputs directory should contain:
- outputs/plots/data_distribution.png
- outputs/plots/correlation_heatmap.png
- outputs/plots/confusion_matrix_decision_tree.png
- outputs/plots/confusion_matrix_random_forest.png
- outputs/plots/model_comparison.png
- outputs/plots/training_history_shallow_nn.png
- outputs/plots/training_history_deep_nn.png
- outputs/plots/feature_importance_random_forest.png
- outputs/plots/pca_variance.png
- outputs/reports/model_comparison.csv
- outputs/models/ (all saved model files)

## Quality Checks After Each Phase

After Phase 1: Run this in Python to verify:
    from src.data.loader import load_excel, display_dataset_info
    from src.data.cleaner import clean_dataset
    df = load_excel("data/restaurante_foodvision.xlsx")
    display_dataset_info(df)
    df_clean, report = clean_dataset(df)
    print(report)

After Phase 2: Verify at least one model trains and evaluates without error on a sample dataset.

After Phase 3: Verify at least one plot saves to outputs/plots/ without error.

After Phase 4: Run python src/main.py and navigate all menu options.

## What You Must Not Do

- Do not generate synthetic data inside the modules (data comes from the Excel file)
- Do not use print statements for debugging - use the logger
- Do not hardcode file paths as strings - use config.py Path objects
- Do not use bare except clauses
- Do not use global variables outside of main.py app_state dict
- Do not skip writing docstrings to save time
- Do not use third-party libraries not in requirements.txt without checking first
- Do not write all modules in one file - maintain the folder structure exactly as specified

## Start Command

Begin now with Phase 1, Module 1: create `src/utils/config.py`.
After completing it, confirm it is done and move to `src/utils/logger.py`.
Continue in sequence until all modules are complete.
