# Standard library imports
import argparse
import logging
import os
import platform
import sys
import warnings

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.exceptions import ConvergenceWarning

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

REQUIRED_CONFIG_KEYS = [
    "file_path", "target_column", "val_test_size", "val_size",
    "ridge_param_grid", "lasso_param_grid", "rf_param_grid", "hgb_param_grid", "cv", "scoring",
    "numerical_features", "nominal_features", "ordinal_features",
]


def setup_logging(log_file: str = None) -> None:
    """
    Configure logging to the console and, optionally, to a file with timestamps and levels.

    Args:
        log_file (str): Path of the log file. If None, only console logging is used.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_config(config_path: str) -> dict:
    """
    Load the YAML configuration file and check that it is valid.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The validated configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML cannot be parsed, keys are missing or values are out of range.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}. Use --config to set its path.")
    except yaml.YAMLError as e:
        raise ValueError(f"Config file {config_path} is not valid YAML: {e}")

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    for key in ["val_test_size", "val_size"]:
        if not 0 < config[key] < 1:
            raise ValueError(f"Config '{key}' must be between 0 and 1, got {config[key]}")
    if config["cv"] < 2:
        raise ValueError(f"Config 'cv' must be at least 2, got {config['cv']}")
    return config


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file and log its shape and missing values.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains no rows.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file is empty: {file_path}")
    if df.empty:
        raise ValueError(f"Data file has no rows: {file_path}")
    missing = df.isna().sum()
    missing = missing[missing > 0].to_dict()
    logging.info(f"Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns. Missing values: {missing or 'none'}")
    return df


def predict(config: dict, input_path: str, model_path: str, output_path: str) -> None:
    """
    Score new students with a saved model, without retraining.

    The new data is cleaned in prediction mode (no rows are removed), so the output has exactly
    one prediction per input row.

    Args:
        config (dict): The validated configuration.
        input_path (str): CSV file with the same columns as the training data, except the target.
        model_path (str): Path to a saved pipeline (best_model.joblib from a previous run).
        output_path (str): CSV file to write the predictions to.
    """
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    df = load_data(input_path)
    student_ids = df["student_id"] if "student_id" in df.columns else pd.Series(df.index, name="row")
    cleaned = DataPreparation(config).clean_data(df, training=False)
    cleaned = cleaned.drop(columns=config["target_column"], errors="ignore")
    if len(cleaned) != len(student_ids):
        raise ValueError("Prediction mode must not remove rows.")
    predictions = pd.DataFrame(
        {student_ids.name or "row": student_ids.values, f"predicted_{config['target_column']}": model.predict(cleaned)}
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logging.info(f"Wrote {len(predictions)} predictions to {output_path}")


def main():
    """
    Run the training pipeline (default) or score new data with a saved model (--predict).

    Training: load and validate the config and data, clean, split, train baseline and tuned
    models, select the best on the validation set, refit it on train + validation, evaluate it
    once on the test set and save the artifacts.
    """
    parser = argparse.ArgumentParser(description="Room price per night prediction pipeline")
    parser.add_argument(
        "--config", default="./src/config.yaml", help="Path to the YAML configuration file"
    )
    parser.add_argument("--predict", metavar="INPUT_CSV", help="Score new students from this CSV instead of training")
    parser.add_argument("--model", help="Saved model (best_model.joblib) to use with --predict")
    parser.add_argument("--output", default="./predictions.csv", help="Where --predict writes its predictions")
    args = parser.parse_args()

    # Hide only sklearn convergence warnings (not every warning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    try:
        config = load_config(args.config)
        setup_logging(config.get("log_file"))
        logging.info(
            f"Environment: python {platform.python_version()}, pandas {pd.__version__}, "
            f"numpy {np.__version__}, scikit-learn {sklearn.__version__}, seed {config.get('random_seed', 42)}"
        )

        if args.predict:
            if not args.model:
                raise ValueError("--predict also needs --model (path to a saved best_model.joblib).")
            predict(config, args.predict, args.model, args.output)
            return

        # Load CSV file into a DataFrame
        df = load_data(config["file_path"])

        # Initialize and run data preparation
        data_prep = DataPreparation(config)
        cleaned_df = data_prep.clean_data(df)

        # Initialize model training with the created preprocessor
        model_training = ModelTraining(config, data_prep.preprocessor)

        # Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(
            cleaned_df
        )

        # Train and evaluate baseline models with default hyperparameters
        baseline_models, baseline_metrics = (
            model_training.train_and_evaluate_baseline_models(
                X_train, y_train, X_val, y_val
            )
        )

        # Train and evaluate tuned models with hyperparameter tuning
        tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
            X_train, y_train, X_val, y_val
        )

        # Combine all models and their metrics into dictionaries
        all_models = {**baseline_models, **tuned_models}
        all_metrics = {**baseline_metrics, **tuned_metrics}
        if not all_metrics:
            raise ValueError("No models were trained.")

        # Find the best model based on R² score on the validation set
        best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["R²"])
        best_model = all_models[best_model_name]
        logging.info(f"Best Model Found: {best_model_name}")

        # Model selection is finished: refit on train + validation, then evaluate once on the test set
        final_model = model_training.refit_on_train_and_validation(
            best_model, X_train, y_train, X_val, y_val
        )
        final_metrics = model_training.evaluate_final_model(
            final_model, X_test, y_test, best_model_name
        )

        # Save the model, metrics, predictions, tuning results and config snapshot
        model_training.save_artifacts(final_model, best_model_name, final_metrics)

    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Pipeline stopped: {e}")
        sys.exit(1)
    except Exception:
        logging.exception("Unexpected error in pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
