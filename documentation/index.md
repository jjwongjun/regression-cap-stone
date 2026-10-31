# Hotel Room Price per Night Prediction: Documentation

This project predicts the price per night of a hotel room (a regression task) from the hotel's
quality, location and booking features. It is a command-line pipeline: one command trains and
compares several models, saves the best one, and a second command scores new rooms with the
saved model.

## Documentation map

| Page | Read it to... |
|---|---|
| [api_reference.md](api_reference.md) | See every module, class, function and command-line option |
| [models.md](models.md) | Understand the data, features, models, results and limitations |
| [configuration.md](configuration.md) | Find out what each `config.yaml` key does and what is allowed |
| [deployment.md](deployment.md) | Install, run, predict on new data, retrain and monitor the model |

## Quick start

```
conda create -n hotelenv python=3.11
conda activate hotelenv
pip install -r requirements.txt

python main.py --test                       # run the tests (about 5 seconds)
python main.py                              # train, compare and save the best model (about 3 minutes)
python main.py --predict new_rooms.csv --model artifacts/run_<timestamp>/best_model.joblib
```

See [deployment.md](deployment.md) for the details, including the format of `new_rooms.csv`.

## How the pipeline works

```
config.yaml ──► load_config ──► load_data (CSV)
                                    │
                    DataPreparation.clean_data
                    (validate columns, clip negative amenities,
                     add 3 engineered features)
                                    │
                         split 80 / 10 / 10
                    train        validation        test
                      │              │               │
      baseline models (6) + tuned models (4, grid search with 5-fold CV)
                      │              │
                      └─ scored on validation ─► best model by R²
                                    │
                    refit best model on train + validation
                                    │
                       one evaluation on the test set
                                    │
          artifacts/run_<timestamp>/  (model, metrics, plots, ranking, config copy)
```

## Project layout

```
main.py                    command-line entry point (train, predict, test)
requirements.txt           pinned dependencies
src/
  config.yaml              all settings (see configuration.md)
  preprocessing.py         DataPreparation: cleaning, imputation, scaling, encoding
  feature_engineering.py   engineered features and feature selection
  model_optimization.py    ModelTraining: splitting, training, tuning, evaluation, saving
data/
  synthetic_data_169.csv   training data (1,000 rows, 8 columns)
artifacts/                 created by each run (one timestamped folder per run)
logs/pipeline.log          created by each run
docs/                      this documentation
```

The file `model_optimization.py` contains the class `ModelTraining` (the file was renamed, the
class was not), so `main.py` imports it as `from src.model_optimization import ModelTraining`.

## Results at a glance

The selected model is a tuned HistGradientBoosting regressor.

| Split | R² | MAE (price per night) | RMSE |
|---|---|---|---|
| Validation (100 rows) | 0.826 | 20.0 | 27.3 |
| Test (100 rows, after refit on train + validation) | 0.661 | 22.5 | 30.7 |

The validation and test sets are small, so these figures are rough. [models.md](models.md)
explains how far to trust them and why the test R² is lower.
