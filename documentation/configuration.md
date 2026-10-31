# Configuration Reference

All settings live in one YAML file, `src/config.yaml` by default. Use another file with
`python main.py --config path/to/config.yaml`. Each run saves a copy of the settings it used to
`config_snapshot.yaml` in its artifacts folder.

## How the file is checked

`load_config` (in `main.py`) stops the run with a clear message if:

- the file is missing or is not valid YAML;
- a required key (marked **required** below) is missing;
- `val_test_size` or `val_size` is not strictly between 0 and 1;
- `cv` is less than 2.

It does **not** check the other values, so a wrong name or a bad grid is reported later, by the
step that uses it.

## Data

| Key | Required | Example | Meaning |
|---|---|---|---|
| `file_path` | **required** | `./data/synthetic_data_169.csv` | CSV with the training data. Relative paths start from where you run `python main.py` (normally the project root). |
| `target_column` | **required** | `room_price_per_night` | Column to predict. |

## Splitting and reproducibility

| Key | Required | Default | Meaning |
|---|---|---|---|
| `random_seed` | optional | `42` | One seed for the splits and for every model that uses randomness. The same seed and data give the same results. |
| `val_test_size` | **required** | 0.2 (example) | Share of the data held out for validation and test together. |
| `val_size` | **required** | 0.5 (example) | Share of the held-out data that becomes the **test** set, not the validation set (the name is misleading). With 0.2 and 0.5 the split is 80% train, 10% validation, 10% test. |

## Outputs

| Key | Required | Meaning |
|---|---|---|
| `artifacts_dir` | optional | Folder where each run creates `run_<YYYYMMDD_HHMMSS>/`. Remove the key to save nothing. |
| `log_file` | optional | Log file written next to the console output. Remove the key for console logging only. |

## Model tuning

| Key | Required | Meaning |
|---|---|---|
| `cv` | **required** | Number of cross-validation folds in the grid searches (at least 2). |
| `scoring` | **required** | Metric the grid searches maximise, as a scikit-learn scorer name (for example `r2`, `neg_mean_absolute_error`). It also names the field in `metrics.json` (`best_cv_r2`). Model selection between models uses validation R² whatever this is set to. |
| `ridge_param_grid` | **required** | Grid for Ridge. |
| `lasso_param_grid` | **required** | Grid for Lasso. |
| `rf_param_grid` | **required** | Grid for RandomForestRegressor (200 trees are fixed in the code). |
| `hgb_param_grid` | **required** | Grid for HistGradientBoostingRegressor. |

Grid rules:

- Every parameter name starts with `regressor__`, because the regressor is the second step of a
  pipeline. `regressor__alpha` means the `alpha` of the model.
- Each parameter holds a **list** of values to try. The search tries every combination, so the
  cost grows quickly: a grid of 3 × 5 × 4 values is 60 fits per fold.
- In YAML write a missing value as `null`, never `None`. A bare `None` is read as the text
  `'None'` and the model rejects it. `True` and `False` are read correctly as booleans.

Current grids and the number of combinations:

| Model | Parameters | Combinations |
|---|---|---|
| Ridge | `alpha`: 0.01, 0.1, 1, 10, 100, 1000; `fit_intercept`: True, False | 12 |
| Lasso | same as Ridge | 12 |
| Random forest | `max_depth`: 5, 10, null; `min_samples_leaf`: 1, 3, 5, 10, 20; `max_features`: 0.33, 0.5, 0.75, 1.0 | 60 |
| HistGradientBoosting | `learning_rate`: 0.02, 0.05, 0.1; `max_depth`: 1, 2, 3; `max_iter`: 200, 400, 600; `min_samples_leaf`: 20, 40, 80 | 81 |

## Features

| Key | Required | Meaning |
|---|---|---|
| `numerical_features` | **required** | Numeric columns. They are imputed with the median and scaled with `RobustScaler`. |
| `nominal_features` | **required** | Unordered categories (`room_type`). One-hot encoded. |
| `ordinal_features` | **required** | Ordered categories (`season`, `location_city_tier`). Encoded as 0, 1, 2 in the order given by the category lists below. |
| `season_categories` | needed by the code* | Order of `season`, lowest to highest: `Low`, `Shoulder`, `High`. |
| `location_city_tier_categories` | needed by the code* | Order of `location_city_tier`, lowest price to highest: `Tier 3`, `Tier 2`, `Tier 1`. |
| `use_feature_engineering` | optional | Default `true`. Adds the three engineered features (see [models.md](models.md)). With `false` the model uses the original columns only. |
| `n_selected_features` | optional | Default `null` (keep every feature). A whole number k keeps only the k highest-scoring features (F-test). Testing showed that k below the total number of features hurts accuracy, so leave it `null` unless you have a reason. |

\* `load_config` does not check these two keys, so if one is missing the run stops later with an
"Unexpected error" and a `KeyError`.

Two things to keep in step when you change features:

- **Order matters.** The code builds the ordinal encoder with the season categories first and the
  city-tier categories second, so `ordinal_features` must list `season` first and
  `location_city_tier` second. Swapping them would apply the wrong order silently.
- **Names are also fixed in code.** `DataPreparation.REQUIRED_FEATURE_COLUMNS` and the engineered
  features in `feature_engineering.py` use fixed column names. Adding or renaming a feature means
  editing the config and those two places.

## Complete example

```yaml
file_path: './data/synthetic_data_169.csv'
target_column: room_price_per_night
random_seed: 42
val_test_size: 0.2
val_size: 0.5
artifacts_dir: './artifacts'
log_file: './logs/pipeline.log'
ridge_param_grid:
  regressor__alpha: [0.01, 0.1, 1, 10, 100, 1000]
  regressor__fit_intercept: [True, False]
lasso_param_grid:
  regressor__alpha: [0.01, 0.1, 1, 10, 100, 1000]
  regressor__fit_intercept: [True, False]
rf_param_grid:
  regressor__max_depth: [5, 10, null]
  regressor__min_samples_leaf: [1, 3, 5, 10, 20]
  regressor__max_features: [0.33, 0.5, 0.75, 1.0]
hgb_param_grid:
  regressor__learning_rate: [0.02, 0.05, 0.1]
  regressor__max_depth: [1, 2, 3]
  regressor__max_iter: [200, 400, 600]
  regressor__min_samples_leaf: [20, 40, 80]
cv: 5
scoring: r2
numerical_features: [hotel_star_rating, distance_from_airport, number_of_amenities, online_rating_score]
nominal_features: [room_type]
ordinal_features: [season, location_city_tier]
season_categories: [Low, Shoulder, High]
location_city_tier_categories: [Tier 3, Tier 2, Tier 1]
use_feature_engineering: true
n_selected_features: null
```

## Common mistakes

| Symptom | Cause and fix |
|---|---|
| `Config is missing required keys: [...]` | Add the keys listed. The grid keys are `ridge_param_grid`, `lasso_param_grid`, `rf_param_grid`, `hgb_param_grid`. |
| `Data file not found` | Wrong `file_path`, or you ran the command from a different folder. |
| `max_depth ... Got 'None' instead` | Use `null` in the YAML, not `None`. |
| `Invalid parameter ...` from a grid search | A grid names a parameter the model does not have, or it is missing the `regressor__` prefix. |
| `KeyError: 'season_categories'` (shown as "Unexpected error") | The category lists are missing. |
