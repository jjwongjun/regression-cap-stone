"""
Feature engineering and feature selection for the hotel room price project.

How it connects to the rest of the project
------------------------------------------
- preprocessing.py calls `create_features` inside `DataPreparation.clean_data`, so the new
  columns exist for training AND for `--predict`. They are computed row by row (no statistics
  are learned from the data), so there is no leakage between the train/validation/test splits.
- preprocessing.py adds `NEW_FEATURES` to the numerical features, so they are imputed and
  scaled like the others. Turn this off with `use_feature_engineering: false` in config.yaml.
- Feature selection has two parts:
    1. `rank_features` scores every model-ready feature with four methods. main.py calls it on
       the training split only and logs/saves the table.
    2. `make_selector` is an optional step inside the preprocessing pipeline. Set
       `n_selected_features: <k>` in config.yaml to keep only the k best features (default:
       keep all). It is fitted inside every cross-validation fold, so it does not leak either.

Where the engineered features come from
---------------------------------------
EDA insights:
- The strongest predictors of price are hotel_star_rating (r = 0.73), online_rating_score
  (r = 0.61) and number_of_amenities (r = 0.57), and they are correlated with each other
  (r = 0.45 to 0.65).
- Price is right-skewed (skew 1.35), which points to effects that multiply rather than add.

Domain knowledge: guests pay a premium for hotels that are good on several quality dimensions
at the same time (a 5-star hotel with great reviews and many amenities), so the price effect of
one quality measure grows with the others. Interaction (product) terms capture this.

Note: hotel_star_rating is already standardised (it has negative values), so only products are
used. Ratios and logs of it would be meaningless.

Tested with 5-fold CV repeated 4 times (R2; linear model / histogram gradient boosting):
    baseline                         0.727 / 0.776
    + the 3 interactions below       0.778 / 0.772   <- kept
    + squares of the numeric columns 0.775 / 0.778   (same gain as the interactions, not additive)
    + season x city tier             0.781 / 0.768   (no real gain, dropped)
    + room_type x star rating        0.780 / 0.771   (no real gain, dropped)
    + log or squared distance, high-season flag, tier-1 flag, far-from-airport flag,
      sum of the quality scores      no gain         (dropped)
The interactions help linear models (+0.05). Tree models already find interactions on their
own, so they are unchanged within noise.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LassoCV

# Columns created by create_features (preprocessing.py adds them to the numerical features)
NEW_FEATURES = ["star_x_online", "star_x_amenities", "amenities_x_online"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the engineered features to a copy of the data.

    Call it after number_of_amenities has been clipped at 0, so impossible values are not used.

    Args:
        df (pd.DataFrame): Cleaned data with the raw feature columns.

    Returns:
        pd.DataFrame: A copy of df with the columns in NEW_FEATURES added.

    Example:
        >>> raw = pd.DataFrame({'hotel_star_rating': [2.0], 'online_rating_score': [3.0],
        ...                     'number_of_amenities': [4.0]})
        >>> create_features(raw)[NEW_FEATURES].values.tolist()
        [[6.0, 8.0, 12.0]]
    """
    df = df.copy()
    df["star_x_online"] = df["hotel_star_rating"] * df["online_rating_score"]
    df["star_x_amenities"] = df["hotel_star_rating"] * df["number_of_amenities"]
    df["amenities_x_online"] = df["number_of_amenities"] * df["online_rating_score"]
    return df


def make_selector(k: int) -> SelectKBest:
    """
    Build a feature selector that keeps the k features with the highest univariate F-score.

    It is meant to sit after the ColumnTransformer inside a Pipeline, so it is fitted on the
    training data only (and on each training fold during cross-validation).
    """
    if k < 1:
        raise ValueError(f"n_selected_features must be at least 1, got {k}")
    return SelectKBest(score_func=f_regression, k=k)


def rank_features(preprocessor, X: pd.DataFrame, y: pd.Series, seed: int = 42) -> pd.DataFrame:
    """
    Score every model-ready feature with four selection methods and rank them.

    - f_score: univariate F-test of a linear relationship with the target (filter method)
    - mutual_info: also catches non-linear relationships with the target (filter method)
    - lasso_coef: absolute Lasso coefficient; weak features shrink to 0 (embedded method)
    - rf_importance: random forest importance, which also reflects interactions (embedded method)

    avg_rank is the mean rank over the four methods (1 = most relevant). Use it to see which
    features carry the signal and which are candidates to drop. Pass the training split only.

    Args:
        preprocessor: An unfitted preprocessor (it is cloned, so the original stays unfitted).
        X (pd.DataFrame): Training features (after cleaning).
        y (pd.Series): Training target.
        seed (int): Random seed.

    Returns:
        pd.DataFrame: One row per feature, sorted from most to least relevant.
    """
    prep = clone(preprocessor)
    X_t = prep.fit_transform(X, y)
    names = [name.split("__", 1)[-1] for name in prep.get_feature_names_out()]

    scores = pd.DataFrame(index=names)
    scores["f_score"] = np.nan_to_num(f_regression(X_t, y)[0])
    scores["mutual_info"] = mutual_info_regression(X_t, y, random_state=seed)
    scores["lasso_coef"] = np.abs(LassoCV(cv=5).fit(X_t, y).coef_)
    scores["rf_importance"] = RandomForestRegressor(
        n_estimators=200, random_state=seed, n_jobs=-1
    ).fit(X_t, y).feature_importances_

    scores["avg_rank"] = scores.rank(ascending=False).mean(axis=1)
    return scores.sort_values("avg_rank")


# ---------------------------------------------------------------------------
# Tests (small synthetic examples)
#
# Run them with any of:
#   python main.py --test
#   python src/feature_engineering.py
#   python -m pytest src/feature_engineering.py
# ---------------------------------------------------------------------------
def _make_test_df() -> pd.DataFrame:
    return pd.DataFrame({
        "hotel_star_rating": [1.0, -2.0, 0.5],
        "online_rating_score": [4.0, 3.0, 5.0],
        "number_of_amenities": [6.0, 0.0, 10.0],
    })


def test_create_features_values_and_columns():
    raw = _make_test_df()
    out = create_features(raw)
    assert list(out.columns) == list(raw.columns) + NEW_FEATURES
    assert out["star_x_online"].tolist() == [4.0, -6.0, 2.5]
    assert out["star_x_amenities"].tolist() == [6.0, -0.0, 5.0]
    assert out["amenities_x_online"].tolist() == [24.0, 0.0, 50.0]


def test_create_features_keeps_rows_and_does_not_change_input():
    raw = _make_test_df()
    out = create_features(raw)
    assert len(out) == len(raw)
    assert list(raw.columns) == ["hotel_star_rating", "online_rating_score", "number_of_amenities"]


def test_rank_features_puts_signal_first_and_noise_last():
    from sklearn.compose import ColumnTransformer
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"signal": rng.normal(size=300), "noise": rng.normal(size=300)})
    y = 100 + 50 * X["signal"] + rng.normal(scale=5, size=300)
    ranking = rank_features(ColumnTransformer([("num", "passthrough", ["signal", "noise"])]), X, y)
    assert list(ranking.index) == ["signal", "noise"]
    assert {"f_score", "mutual_info", "lasso_coef", "rf_importance", "avg_rank"} <= set(ranking.columns)


def test_make_selector_keeps_k_features():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    y = 3 * X[:, 0] + rng.normal(size=100)
    selected = make_selector(2).fit(X, y)
    assert selected.transform(X).shape == (100, 2)
    assert selected.get_support()[0]   # the informative column is kept
    try:
        make_selector(0)
    except ValueError:
        pass
    else:
        raise AssertionError("make_selector(0) should raise a ValueError")


def run_tests() -> None:
    """Run every test_* function in this module and print the result (no pytest needed)."""
    tests = [f for name, f in globals().items() if name.startswith("test_") and callable(f)]
    for test in tests:
        test()
        print(f"PASSED  {test.__name__}")
    print(f"All {len(tests)} feature engineering tests passed.")


if __name__ == "__main__":
    run_tests()
