"""
Advanced regression models for the hotel room price project: Random Forest and Gradient Boosting.

The baseline models (dummy, linear regression, Ridge, Lasso) are in model_optimization.py. The two
models here are different architectures that can capture non-linear effects and interactions:

- RandomForestModel: many deep decision trees built independently on random samples and averaged
  (bagging). Robust to outliers and needs little tuning.
- GradientBoostingModel: shallow trees built one after another, each correcting the errors of the
  previous ones (boosting; scikit-learn's histogram version). Usually the strongest on tabular data.

Both follow the same interface, so they are created, trained and used in exactly the same way:

    model = RandomForestModel(config, preprocessor)
    model.fit(X_train, y_train)          # grid search with cross-validation, using the grid in config.yaml
    predictions = model.predict(X_val)

After fit, `model.pipeline_` is the best fitted pipeline (preprocessing + regressor),
`model.best_params_` the chosen parameters and `model.best_cv_score_` their cross-validation score.

main.py creates the two models, ModelTraining.train_and_evaluate_advanced_models trains and
evaluates them, and they are compared with the baseline models in the same table.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class AdvancedModel:
    """
    Base class: one regressor, tuned with grid search, behind a fit / predict interface.

    Subclasses set `name` (the name used in the results), `grid_key` (the config key holding the
    parameter grid) and implement `build_regressor`.

    Attributes:
    -----------
    pipeline_ : sklearn.pipeline.Pipeline
        The best fitted pipeline (None until fit is called).
    best_params_ : dict
        The parameters chosen by the grid search.
    best_cv_score_ : float
        The cross-validation score (config 'scoring') of the best parameters.
    search_ : sklearn.model_selection.GridSearchCV
        The fitted grid search, with the full cross-validation results.
    """

    name: str = ""
    grid_key: str = ""

    def __init__(self, config: Dict[str, Any], preprocessor):
        """
        Args:
        -----
        config (Dict[str, Any]): The configuration (needs 'cv', 'scoring' and the grid under grid_key).
        preprocessor: An unfitted preprocessor; each model works on its own copy of it.
        """
        self.config = config
        self.preprocessor = preprocessor
        self.seed = config.get("random_seed", 42)
        self.pipeline_ = None
        self.best_params_ = None
        self.best_cv_score_ = None
        self.search_ = None

    def build_regressor(self):
        """Return an unfitted scikit-learn regressor (implemented by each model)."""
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AdvancedModel":
        """Tune the model with grid search and cross-validation on (X, y); returns self."""
        pipeline = Pipeline(steps=[
            ("preprocessor", clone(self.preprocessor)),
            ("regressor", self.build_regressor()),
        ])
        search = GridSearchCV(
            pipeline, self.config[self.grid_key], cv=self.config["cv"], scoring=self.config["scoring"],
            n_jobs=-1, error_score="raise",  # a failed fit must stop the run, never be scored as NaN
        )
        search.fit(X, y)
        self.search_ = search
        self.pipeline_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_cv_score_ = float(search.best_score_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with the best fitted pipeline. Raises ValueError if fit was not called."""
        if self.pipeline_ is None:
            raise ValueError(f"{self.name} has not been fitted yet. Call fit first.")
        return self.pipeline_.predict(X)


class RandomForestModel(AdvancedModel):
    """Random forest (200 trees); tuned with `rf_param_grid` from the config."""

    name = "random_forest"
    grid_key = "rf_param_grid"

    def build_regressor(self):
        return RandomForestRegressor(n_estimators=200, random_state=self.seed)


class GradientBoostingModel(AdvancedModel):
    """Histogram gradient boosting; tuned with `hgb_param_grid` from the config."""

    name = "hist_gradient_boosting"
    grid_key = "hgb_param_grid"

    def build_regressor(self):
        return HistGradientBoostingRegressor(random_state=self.seed)


# ---------------------------------------------------------------------------
# Tests (small synthetic examples)
#
# Run them with any of:
#   python main.py --test
#   python src/advanced_models.py
#   python -m pytest src/advanced_models.py
# ---------------------------------------------------------------------------
def _make_test_data():
    from sklearn.compose import ColumnTransformer
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    y = 100 + 30 * X["a"] + 10 * X["b"] ** 2 + rng.normal(scale=2, size=120)
    config = {
        "cv": 2, "scoring": "r2", "random_seed": 0,
        "rf_param_grid": {"regressor__max_depth": [2, 4]},
        "hgb_param_grid": {"regressor__max_depth": [1, 2], "regressor__max_iter": [20, 40]},
    }
    return config, ColumnTransformer([("num", "passthrough", ["a", "b"])]), X, y


def test_both_models_share_the_same_interface():
    config, preprocessor, X, y = _make_test_data()
    for model_class in (RandomForestModel, GradientBoostingModel):
        model = model_class(config, preprocessor).fit(X, y)
        assert isinstance(model.pipeline_, Pipeline)
        assert model.predict(X).shape == (120,)
        assert np.isfinite(model.best_cv_score_)


def test_models_are_distinct_and_use_their_own_grid():
    config, preprocessor, X, y = _make_test_data()
    assert RandomForestModel.name != GradientBoostingModel.name
    rf = RandomForestModel(config, preprocessor).fit(X, y)
    gb = GradientBoostingModel(config, preprocessor).fit(X, y)
    assert rf.best_params_["regressor__max_depth"] in [2, 4]
    assert set(gb.best_params_) == {"regressor__max_depth", "regressor__max_iter"}
    assert type(rf.pipeline_.named_steps["regressor"]) is RandomForestRegressor
    assert type(gb.pipeline_.named_steps["regressor"]) is HistGradientBoostingRegressor


def test_predict_before_fit_raises():
    config, preprocessor, X, y = _make_test_data()
    try:
        RandomForestModel(config, preprocessor).predict(X)
    except ValueError as e:
        assert "not been fitted" in str(e)
    else:
        raise AssertionError("predict before fit should raise a ValueError")


def test_the_original_preprocessor_stays_unfitted():
    config, preprocessor, X, y = _make_test_data()
    GradientBoostingModel(config, preprocessor).fit(X, y)
    assert not hasattr(preprocessor, "transformers_")   # each model fitted its own copy


def run_tests() -> None:
    """Run every test_* function in this module and print the result (no pytest needed)."""
    tests = [f for name, f in globals().items() if name.startswith("test_") and callable(f)]
    for test in tests:
        test()
        print(f"PASSED  {test.__name__}")
    print(f"All {len(tests)} advanced model tests passed.")


if __name__ == "__main__":
    run_tests()
