# Standard library imports
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Tuple

# Related third-party imports
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline


class ModelTraining:
    """
    A class used to train and evaluate machine learning models that predict room price per night.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    """

    # Metrics every evaluation must report (checked in _compute_metrics_and_plots)
    REQUIRED_METRICS = ["MAE", "MSE", "RMSE", "R²"]

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialize the ModelTraining class with configuration and preprocessor.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        preprocessor (sklearn.compose.ColumnTransformer): A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
        """
        self.config = config
        self.preprocessor = preprocessor
        self.seed = config.get("random_seed", 42)
        # Each run saves its outputs to its own timestamped folder so runs can be compared
        artifacts_dir = config.get("artifacts_dir")
        self.run_dir = (
            os.path.join(artifacts_dir, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
            if artifacts_dir else None
        )
        self.tuning_results: Dict[str, Dict[str, Any]] = {}  # best params and CV score per tuned model
        self.test_predictions = None  # predictions of the last test-set evaluation

    def _new_pipeline(self, model) -> Pipeline:
        """
        Build a pipeline with its own copy of the preprocessor.

        The preprocessor is cloned so that fitted state from one model can never leak into another.

        Args:
        -----
        model: An unfitted scikit-learn regressor.

        Returns:
        --------
        Pipeline: Preprocessor followed by the regressor.
        """
        return Pipeline(steps=[("preprocessor", clone(self.preprocessor)), ("regressor", model)])

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the data into training, validation, and test sets.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the cleaned data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: A tuple containing the training, validation, and test features and target variables.

        Raises:
        -------
        ValueError: If the target column is missing from the data.
        """
        logging.info("Starting data splitting.")
        target = self.config["target_column"]
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in the data.")
        X = df.drop(columns=target)
        y = df[target]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.config["val_test_size"], random_state=self.seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config["val_size"], random_state=self.seed
        )
        self._check_split_sizes(X_train, X_val, X_test)

        # Log the target distribution of every split (should be similar if the split is fair)
        self._log_target_distribution(y, "Full dataset")
        self._log_target_distribution(y_train, "Train")
        self._log_target_distribution(y_val, "Validation")
        self._log_target_distribution(y_test, "Test")

        logging.info("Data split into train, validation, test sets.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train, and evaluate baseline models.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.")
        self._check_preprocessing(X_train)
        models = {
            "dummy_mean": DummyRegressor(strategy="mean"),  # trivial benchmark every real model must beat
            "linear_regression": LinearRegression(),
            "ridge": Ridge(),
            "lasso": Lasso(),
            "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=self.seed),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=self.seed),
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            pipeline = self._new_pipeline(model)
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline
            metrics[model_name] = self._evaluate_model(
                pipeline, X_val, y_val, model_name
            )

        return pipelines, metrics

    def train_and_evaluate_tuned_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform hyperparameter tuning (grid search with cross-validation) for Ridge, Lasso, Random Forest, HistGradientBoosting models and evaluate them.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the tuned pipelines and their evaluation metrics.
        """
        logging.info("Starting hyperparameter tuning.")
        tuned_models = {}
        tuned_metrics = {}
        cv = self.config["cv"]
        scoring = self.config["scoring"]

        models = {
       "ridge_tuned": (Ridge(), self.config["ridge_param_grid"]),
       "lasso_tuned": (Lasso(), self.config["lasso_param_grid"]),
       "rf_tuned": (RandomForestRegressor(n_estimators=200, random_state=self.seed), self.config["rf_param_grid"]),
       "hgb_tuned": (HistGradientBoostingRegressor(random_state=self.seed), self.config["hgb_param_grid"]),
        }
        for model_name, (model, param_grid) in models.items():
            pipeline = self._new_pipeline(model)
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1,
                error_score="raise"  # a failed fit must stop the run, never be scored as NaN silently
            )
            grid_search.fit(X_train, y_train)
            tuned_models[model_name] = grid_search.best_estimator_
            self.tuning_results[model_name] = {
                "best_params": grid_search.best_params_,
                f"best_cv_{scoring}": float(grid_search.best_score_),
            }
            logging.info(
                f"{model_name}: best params {grid_search.best_params_}, "
                f"best CV {scoring} = {grid_search.best_score_:.4f}"
            )
            self._save_cv_results(grid_search, model_name)
            tuned_metrics[model_name] = self._evaluate_model(
                tuned_models[model_name], X_val, y_val, model_name
            )

        logging.info("Hyperparameter tuning completed.")
        return tuned_models, tuned_metrics

    def refit_on_train_and_validation(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Pipeline:
        """
        Refit the selected model on the training and validation data combined.

        Model selection is finished at this point, so the validation data no longer needs to be held
        back: using it for training gives the final model more data before the one-time test evaluation.
        The hyperparameters chosen earlier are kept, only the model's fitted state is rebuilt.

        Args:
        -----
        model (Pipeline): The selected model pipeline.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Pipeline: A new pipeline fitted on training + validation data.
        """
        refitted = clone(model)
        refitted.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        logging.info(f"Refitted the selected model on train + validation data ({len(X_train) + len(X_val)} rows).")
        return refitted

    def evaluate_final_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate the final model on the test set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        metrics = self._compute_metrics_and_plots(
            model, X_test, y_test, f"{model_name} (test)", save_plots=True
        )
        logging.info(f"Final Test Metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics

    def _evaluate_model(
        self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a model on the validation set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        metrics = self._compute_metrics_and_plots(model, X_val, y_val, model_name)
        logging.info(f"{model_name} Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics

    def _compute_metrics_and_plots(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        save_plots: bool = False,
    ) -> Dict[str, float]:
        """
        Compute regression metrics and draw the predicted-vs-actual and residual plots.

        Shared by the validation and test evaluations so the metrics and plots stay identical.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X (pd.DataFrame): The features to evaluate on.
        y (pd.Series): The true target values.
        model_name (str): Name used in plot titles and file names.
        save_plots (bool): Whether to save the plots (and keep the predictions) for the artifacts.

        Returns:
        --------
        Dict[str, float]: MAE, MSE, RMSE and R².
        """
        y_pred = model.predict(X)
        metrics = {
            "MAE": mean_absolute_error(y, y_pred),
            "MSE": mean_squared_error(y, y_pred),
            "RMSE": root_mean_squared_error(y, y_pred),
            "R²": r2_score(y, y_pred),
        }
        missing_keys = [k for k in self.REQUIRED_METRICS if k not in metrics]
        if missing_keys:
            raise ValueError(f"Metrics are missing required keys: {missing_keys}")

        residuals = np.asarray(y) - y_pred
        fig, (ax_fit, ax_res, ax_hist) = plt.subplots(1, 3, figsize=(17, 5))
        ax_fit.scatter(y, y_pred, alpha=0.3)
        lims = [min(np.min(y), np.min(y_pred)), max(np.max(y), np.max(y_pred))]
        ax_fit.plot(lims, lims, color='red', linestyle='--', label='Perfect prediction')
        ax_fit.set_xlabel('Actual room price')
        ax_fit.set_ylabel('Predicted room price')
        ax_fit.set_title(f"{model_name} - Predicted vs Actual (R² = {metrics['R²']:.3f})")
        ax_fit.legend()
        ax_res.scatter(y_pred, residuals, alpha=0.3)
        ax_res.axhline(0, color='red', linestyle='--')
        ax_res.set_xlabel('Predicted score')
        ax_res.set_ylabel('Residual (actual - predicted)')
        ax_res.set_title(f"{model_name} - Residuals vs Predicted")
        ax_hist.hist(residuals, bins=40)
        ax_hist.set_xlabel('Residual (actual - predicted)')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title(f"{model_name} - Residual distribution")
        fig.tight_layout()
        self._finish_plot(fig, f"{model_name}_diagnostics", save_plots)

        if save_plots:
            self.test_predictions = pd.DataFrame(
                {"actual": np.asarray(y), "predicted": y_pred, "residual": residuals}, index=X.index
            )
        return metrics

    def _finish_plot(self, fig: plt.Figure, name: str, save: bool) -> None:
        """
        Optionally save a figure to the run folder, then show and close it.

        Args:
        -----
        fig (plt.Figure): The figure to finish.
        name (str): Base file name (without extension).
        save (bool): Whether to save the figure.
        """
        if save and self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            fig.savefig(os.path.join(self.run_dir, f"{safe_name}.png"), bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def _save_cv_results(self, grid_search: GridSearchCV, model_name: str) -> None:
        """
        Save the full cross-validation results of a grid search so the model selection can be audited.

        Args:
        -----
        grid_search (GridSearchCV): A fitted grid search.
        model_name (str): Name used in the file name.
        """
        if not self.run_dir:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        columns = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        pd.DataFrame(grid_search.cv_results_)[columns].sort_values("rank_test_score").to_csv(
            os.path.join(self.run_dir, f"cv_results_{model_name}.csv"), index=False
        )

    def save_artifacts(
        self,
        model: Pipeline,
        model_name: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Save the selected model, its test metrics and predictions, tuning results and a config snapshot.

        Files are written to a timestamped folder inside the 'artifacts_dir' set in the config.
        Nothing is saved if it is not set.

        Args:
        -----
        model (Pipeline): The selected trained model pipeline.
        model_name (str): The name of the selected model.
        metrics (Dict[str, float]): The test-set metrics of the selected model.
        """
        if not self.run_dir:
            return
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            joblib.dump(model, os.path.join(self.run_dir, "best_model.joblib"))
            summary = {
                "model_name": model_name,
                "random_seed": self.seed,
                "test_metrics": {k: float(v) for k, v in metrics.items()},
                "tuning_results": self.tuning_results,
            }
            with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
                json.dump(summary, f, indent=2)
            with open(os.path.join(self.run_dir, "config_snapshot.yaml"), "w") as f:
                yaml.safe_dump(self.config, f)
            if self.test_predictions is not None:
                self.test_predictions.to_csv(os.path.join(self.run_dir, "test_predictions.csv"))
            logging.info(f"Artifacts saved to {self.run_dir}")
        except OSError:
            logging.exception("Failed to save artifacts.")
            raise

    @staticmethod
    def _check_split_sizes(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> None:
        """
        Sanity check that the splits are non-empty and that the training set is the largest.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        X_val (pd.DataFrame): The validation features.
        X_test (pd.DataFrame): The test features.

        Raises:
        -------
        ValueError: If a split is empty or the training set is not larger than validation and test.

        Example:
            >>> big, small = pd.DataFrame({'a': range(8)}), pd.DataFrame({'a': range(1)})
            >>> ModelTraining._check_split_sizes(big, small, small)
            >>> ModelTraining._check_split_sizes(small, big, big)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: Training set must be larger than the validation and test sets.
        """
        if min(len(X_train), len(X_val), len(X_test)) == 0:
            raise ValueError("Train, validation and test sets must all be non-empty.")
        if not len(X_train) > max(len(X_val), len(X_test)):
            raise ValueError("Training set must be larger than the validation and test sets.")

    def _check_preprocessing(self, X_train: pd.DataFrame) -> None:
        """
        Sanity check that the preprocessor produces only finite numbers (no NaN left after imputation).

        A copy of the preprocessor is fitted, so the real one is untouched.

        Args:
        -----
        X_train (pd.DataFrame): The training features.

        Raises:
        -------
        ValueError: If the preprocessed training data contains NaN or infinite values.
        """
        transformed = clone(self.preprocessor).fit_transform(X_train)
        if not np.isfinite(np.asarray(transformed, dtype=float)).all():
            raise ValueError("Preprocessed training data still contains NaN or infinite values.")
        logging.info(f"Preprocessing check passed: {transformed.shape[1]} features after encoding.")

    @staticmethod
    def _log_target_distribution(y_split: pd.Series, name: str) -> None:
        """
        Log summary statistics of the target for one data split.

        Args:
        -----
        y_split (pd.Series): The target values of the split.
        name (str): Name of the split, used in the log message.
        """
        stats = y_split.describe()
        logging.info(
            f"--- {name} (n={len(y_split)}) --- target mean {stats['mean']:.2f}, std {stats['std']:.2f}, "
            f"min {stats['min']:.0f}, median {stats['50%']:.0f}, max {stats['max']:.0f}"
        )
