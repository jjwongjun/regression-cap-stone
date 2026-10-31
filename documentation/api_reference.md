# API Reference

Modules, in the order the pipeline uses them. Names starting with an underscore are internal
helpers; they are listed briefly because other code and the tests rely on them.

Contents: [Command line](#command-line) · [main.py](#mainpy) · [preprocessing.py](#preprocessingpy) ·
[feature_engineering.py](#feature_engineeringpy) · [model_optimization.py](#model_optimizationpy) ·
[Files written by a run](#files-written-by-a-run) · [Errors and exit codes](#errors-and-exit-codes)

## Command line

Run everything from the project root.

| Command | What it does |
|---|---|
| `python main.py` | Train, tune, compare, select, refit, test and save (the default). |
| `python main.py --config PATH` | Use another configuration file (default `./src/config.yaml`). |
| `python main.py --predict INPUT.csv --model MODEL.joblib` | Score new rooms with a saved model. |
| `python main.py --predict INPUT.csv --model MODEL.joblib --output OUT.csv` | Same, choosing where the predictions go (default `./predictions.csv`). |
| `python main.py --test` | Run the preprocessing and feature engineering tests and exit. |

`--predict` needs `--model`. `--test` needs no config or data file.

## main.py

The entry point. It only coordinates the other modules.

### `setup_logging(log_file: str = None) -> None`
Logs to the console and, if `log_file` is given, to that file (the folder is created). Format:
`timestamp | LEVEL | message`.

### `load_config(config_path: str) -> dict`
Reads the YAML file and checks it (see [configuration.md](configuration.md)). Raises
`FileNotFoundError` if the file is missing and `ValueError` for invalid YAML, missing keys or
out-of-range values.

### `load_data(file_path: str) -> pd.DataFrame`
Reads a CSV and logs its shape and any missing values. Raises `FileNotFoundError` if the file does
not exist and `ValueError` if it is empty.

### `predict(config: dict, input_path: str, model_path: str, output_path: str) -> None`
Scores new rooms with a saved model, without retraining.

1. Loads the model with `joblib.load` and the CSV with `load_data`.
2. Cleans the rows with `DataPreparation.clean_data(df, training=False)`, which adds the
   engineered features and never removes rows.
3. Writes a CSV with one row per input row: a `row` column (the position of the row in the input)
   and `predicted_<target>` (for example `predicted_room_price_per_night`).

Raises `FileNotFoundError` if the model file is missing and `ValueError` if a row would be
dropped. The input does not need the target column. See [deployment.md](deployment.md) for the
input format.

### `main() -> None`
Parses the command line and runs one of three things: the tests, `predict`, or the training
pipeline:

1. Load and validate the config, set up logging, log the library versions and seed.
2. Load the data and clean it (`DataPreparation`).
3. Split into train, validation and test (`ModelTraining.split_data`).
4. Rank the features on the training split and save `feature_ranking.csv`.
5. Train the baseline models, then the tuned models; collect their validation metrics.
6. Select the model with the highest validation R².
7. Refit it on train + validation and evaluate it once on the test set.
8. Save the artifacts.

Any `FileNotFoundError` or `ValueError` is logged as `Pipeline stopped: ...`; any other error is
logged with its traceback. Both exit with status 1.

## preprocessing.py

### `class DataPreparation(config: dict)`
Cleans the raw data and builds the preprocessor that every model uses.

Attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `config` | dict | The configuration. |
| `use_feature_engineering` | bool | From `use_feature_engineering` (default `True`). |
| `column_transformer` | `ColumnTransformer` | Imputation, scaling and encoding of all features, without any selection step. |
| `preprocessor` | `ColumnTransformer` or `Pipeline` | What the models use. Equal to `column_transformer`, or a `Pipeline` that adds a `SelectKBest` step when `n_selected_features` is set. |

Class constant `REQUIRED_FEATURE_COLUMNS`: the seven raw feature columns that must be in the data
(`room_type`, `location_city_tier`, `season`, `hotel_star_rating`, `distance_from_airport`,
`number_of_amenities`, `online_rating_score`).

What the preprocessor does to each feature group:

| Group | Steps |
|---|---|
| Numerical (the four raw columns plus the three engineered ones) | median imputation, then `RobustScaler` (centre on the median, scale by the interquartile range, so outliers cannot distort the scaling) |
| Nominal (`room_type`) | most-frequent imputation, then one-hot encoding; an unseen category becomes all zeros |
| Ordinal (`season`, `location_city_tier`) | most-frequent imputation, then ordinal encoding in the order of the category lists; an unseen category becomes -1 |
| Any other column | dropped (`remainder="drop"`), so an unexpected column never reaches a model |

The output is always a dense array. All fitted values (medians, scales, categories) come from the
training data only.

#### `clean_data(df: pd.DataFrame, training: bool = True) -> pd.DataFrame`
Returns a cleaned copy; the input is not changed and no rows are removed.

- Checks that the data is not empty and has the required columns (plus the target when
  `training=True`).
- Sets negative `number_of_amenities` values to 0.
- Adds the engineered features (if `use_feature_engineering`).
- Checks that every configured feature is present and, for training data, that the target has no
  missing values.

Raises `ValueError` if a check fails. Use `training=False` for new data without a target.

#### `fit(df, y=None)`, `transform(df)`, `fit_transform(df, y=None)`
Use the preprocessor on its own. `fit` returns `self`. `y` is only needed when feature selection
is on. Pass cleaned data (the output of `clean_data`).

#### Internal helpers
`_validate_schema(df, required_columns)` (static; empty-data and missing-column checks, has
doctests), `_check_cleaned_data(df, training)`, `_create_preprocessor()`.

### `run_tests()`
Runs the ten tests at the bottom of the file (`test_*` functions on small synthetic data): output
shape, ordinal mapping, unseen categories, missing values, clipping, schema errors, switching
feature engineering off, feature selection, and an end-to-end flow. Also works with
`python src/preprocessing.py` and `python -m pytest src/preprocessing.py`.

## feature_engineering.py

### `NEW_FEATURES`
`["star_x_online", "star_x_amenities", "amenities_x_online"]`: the columns added by
`create_features`.

### `create_features(df: pd.DataFrame) -> pd.DataFrame`
Returns a copy of `df` with three product features:

| Column | Formula |
|---|---|
| `star_x_online` | `hotel_star_rating` × `online_rating_score` |
| `star_x_amenities` | `hotel_star_rating` × `number_of_amenities` |
| `amenities_x_online` | `number_of_amenities` × `online_rating_score` |

They are computed row by row, so nothing is learned from the data and there is no leakage
between splits. Call it after the amenities have been clipped at 0 (`clean_data` does this).

### `make_selector(k: int) -> SelectKBest`
A selector that keeps the `k` features with the highest univariate F-score. Raises `ValueError`
if `k < 1`. It is fitted inside the pipeline, so it only ever sees training data.

### `rank_features(preprocessor, X, y, seed=42) -> pd.DataFrame`
Scores every model-ready feature with four methods and ranks them. The preprocessor is cloned, so
the original stays unfitted. Pass training data only.

| Column | Method |
|---|---|
| `f_score` | Univariate F-test (linear relationship with the target) |
| `mutual_info` | Mutual information (also catches non-linear relationships) |
| `lasso_coef` | Absolute `LassoCV` coefficient (0 means Lasso dropped the feature) |
| `rf_importance` | Random forest importance (reflects interactions) |
| `avg_rank` | Mean rank over the four methods; 1 is most relevant |

The result is sorted by `avg_rank`.

### `run_tests()`
Runs the four tests in the file (feature values, no change to the input, ranking, selector).
Also `python src/feature_engineering.py` or `python -m pytest src/feature_engineering.py`.

## model_optimization.py

### `class ModelTraining(config: dict, preprocessor)`
Trains, tunes, evaluates and saves the models. `preprocessor` is `DataPreparation.preprocessor`.
Every model gets its own clone of it, so fitted values never pass from one model to another.

| Attribute | Meaning |
|---|---|
| `seed` | `random_seed` from the config (default 42). |
| `run_dir` | `<artifacts_dir>/run_<timestamp>`, or `None` if `artifacts_dir` is not set. |
| `tuning_results` | Best parameters and best CV score of each tuned model. |
| `test_predictions` | Predictions of the last test evaluation. |
| `REQUIRED_METRICS` | `["MAE", "MSE", "RMSE", "R²"]`. |

#### `split_data(df) -> (X_train, X_val, X_test, y_train, y_val, y_test)`
Random split using `val_test_size`, `val_size` and the seed, then checks that no split is empty
and the training set is the largest. Logs the target's mean, spread and range in each split.
Raises `ValueError` if the target column is missing.

#### `train_and_evaluate_baseline_models(X_train, y_train, X_val, y_val) -> (models, metrics)`
First checks that the preprocessor gives only finite numbers. Then trains six models with default
settings and returns them with their validation metrics (names as keys):
`dummy_mean`, `linear_regression`, `ridge`, `lasso`, `random_forest` (200 trees, minimum 5 samples
per leaf), `hist_gradient_boosting`.

#### `train_and_evaluate_tuned_models(X_train, y_train, X_val, y_val) -> (models, metrics)`
Grid search with cross-validation (`cv`, `scoring`) for `ridge_tuned`, `lasso_tuned`, `rf_tuned`
(200 trees) and `hgb_tuned`, using the grids in the config. A failed fit stops the run instead of
being scored as missing. Stores the best parameters and score in `tuning_results` and saves the
full grid-search results.

#### `refit_on_train_and_validation(model, X_train, y_train, X_val, y_val) -> Pipeline`
Clones the selected model (keeping its parameters) and refits it on train + validation.

#### `evaluate_final_model(model, X_test, y_test, model_name) -> dict`
Computes the metrics on the test set, saves the diagnostic plot and keeps the predictions. Call it
once, after model selection.

#### `save_artifacts(model, model_name, metrics) -> None`
Writes the files listed below. Does nothing if `artifacts_dir` is not set.

#### Internal helpers
`_new_pipeline(model)`, `_evaluate_model(...)`, `_compute_metrics_and_plots(...)` (metrics and the
three diagnostic plots), `_finish_plot(...)`, `_save_cv_results(...)`, `_check_split_sizes(...)`
(static, has doctests), `_check_preprocessing(X_train)`, `_log_target_distribution(...)` (static).

The metrics are MAE, MSE, RMSE (all in price units, or squared) and R² (share of price variance
explained).

## Files written by a run

In `artifacts/run_<timestamp>/`:

| File | Content |
|---|---|
| `best_model.joblib` | The selected model as one pipeline: preprocessing plus regressor. Use it with `--predict`. |
| `metrics.json` | Selected model's name, seed, test metrics, and each tuned model's best parameters and CV score |
| `test_predictions.csv` | Actual price, predicted price and residual for each test row |
| `<model>_test_diagnostics.png` | Predicted vs actual, residuals vs predicted, residual histogram |
| `cv_results_<model>.csv` | Every grid-search combination with its CV score and rank (four files) |
| `feature_ranking.csv` | Output of `rank_features` |
| `config_snapshot.yaml` | The configuration used |

The log is written to `log_file` (default `logs/pipeline.log`).

## Errors and exit codes

| Exit status | Meaning |
|---|---|
| 0 | The run finished (or the tests passed). |
| 1 | A `FileNotFoundError`, a `ValueError` (bad config, missing columns, empty data) or an unexpected error. The reason is in the log. |
