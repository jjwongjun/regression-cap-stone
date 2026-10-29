# Standard library imports
import logging
from typing import Any, Dict, List

# Related third-party imports
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder


class DataPreparation:
    """
    A class used to clean and preprocess data for predicting hotel room price per night.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal and ordinal features.
    """

    # Raw columns the cleaning and feature engineering rely on (needed for training and for prediction)
    REQUIRED_FEATURE_COLUMNS = [
        "room_type", "location_city_tier", "season", "hotel_star_rating",
        "distance_from_airport", "number_of_amenities", "online_rating_score",
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Negative values of number_of_amenities (impossible counts) are set to 0. No rows are removed,
        so every input row is kept. With training=True (the default) the target column must also be
        present and free of missing values; with training=False (used when predicting for new hotel
        rooms) the target is not needed.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.
        training (bool): Whether the data is used for training (True) or prediction (False).

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        target = self.config["target_column"]
        required = self.REQUIRED_FEATURE_COLUMNS + ([target] if training else [])
        self._validate_schema(df, required)
        logging.info(f"Raw data has {df.shape[0]} rows and {df.shape[1]} columns.")
        df = df.copy()
        df['number_of_amenities'] = df['number_of_amenities'].clip(lower=0)  
        self._check_cleaned_data(df, training)
        logging.info(f"Data cleaning completed. Cleaned data has {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

    def fit(self, df: pd.DataFrame) -> "DataPreparation":
        """Fit the preprocessor on (cleaned) training data and return self."""
        self.preprocessor.fit(df)
        return self

    def transform(self, df: pd.DataFrame):
        """Transform (cleaned) data with the fitted preprocessor into a model-ready array."""
        return self.preprocessor.transform(df)

    def fit_transform(self, df: pd.DataFrame):
        """Fit the preprocessor on (cleaned) training data and return the transformed array."""
        return self.preprocessor.fit_transform(df)

    @staticmethod
    def _validate_schema(df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        Checks that the data is not empty and has the columns the pipeline relies on.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame.
        required_columns (List[str]): Column names that must be present.

        Raises:
        -------
        ValueError: If the data is empty or any required column is missing.

        Example:
            >>> DataPreparation._validate_schema(pd.DataFrame({'a': [1]}), ['a'])
            >>> DataPreparation._validate_schema(pd.DataFrame({'a': [1]}), ['a', 'b'])  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: Input data is missing required columns: ['b']
            >>> DataPreparation._validate_schema(pd.DataFrame({'a': []}), ['a'])  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: Input data is empty.
        """
        if df.empty:
            raise ValueError("Input data is empty.")
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}")

    def _check_cleaned_data(self, df: pd.DataFrame, training: bool) -> None:
        """
        Sanity checks run after cleaning, so problems are caught before splitting and training.

        Checks that the data is not empty, that every configured feature is present and, for training
        data, that the target has no missing values.

        Args:
        -----
        df (pd.DataFrame): The cleaned DataFrame.
        training (bool): Whether the data is used for training.

        Raises:
        -------
        ValueError: If any check fails.
        """
        if df.empty:
            raise ValueError("No rows are left after cleaning.")
        features = (
            self.config["numerical_features"]
            + self.config["nominal_features"]
            + self.config["ordinal_features"]
        )
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise ValueError(f"Cleaned data is missing expected feature columns: {missing}")
        if training:
            target = df[self.config["target_column"]]
            if target.isna().any():
                raise ValueError("Cleaned target must not contain missing values.")

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical and nominal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        # Impute first, then scale. The median is used because it is not affected by extreme values.
        # RobustScaler centres on the median and scales by the IQR, so outliers (for example the
        # far-away hotels in distance_from_airport) cannot distort the scaling. Both are learned
        # from the training data only.
        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        nominal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        ordinal_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=[
                            self.config["season_categories"],
                            self.config["location_city_tier_categories"],
                        ],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                )
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("ord", ordinal_transformer, self.config["ordinal_features"]),
            ],
            remainder="drop",  # any column that is not listed above must never reach the model silently
            sparse_threshold=0,  # always return a dense array so every model type can consume it
            n_jobs=-1,
        )
        return preprocessor


# ---------------------------------------------------------------------------
# Tests for the preprocessing (small synthetic examples)
#
# Run them with any of:
#   python main.py --test
#   python src/preprocessing.py
#   python -m pytest src/preprocessing.py
# ---------------------------------------------------------------------------
TEST_CONFIG = {
    "target_column": "room_price_per_night",
    "numerical_features": ["hotel_star_rating", "distance_from_airport",
                           "number_of_amenities", "online_rating_score"],
    "nominal_features": ["room_type"],
    "ordinal_features": ["season", "location_city_tier"],
    "season_categories": ["Low", "Shoulder", "High"],
    "location_city_tier_categories": ["Tier 3", "Tier 2", "Tier 1"],
}


def _make_test_df() -> pd.DataFrame:
    """Six raw rows with every category present and one impossible (negative) amenity count."""
    return pd.DataFrame({
        "room_type": ["Standard", "Deluxe", "Suite", "Standard", "Deluxe", "Suite"],
        "location_city_tier": ["Tier 1", "Tier 2", "Tier 3", "Tier 1", "Tier 2", "Tier 3"],
        "season": ["Low", "Shoulder", "High", "Low", "Shoulder", "High"],
        "hotel_star_rating": [0.5, -0.2, 1.1, -1.0, 0.3, 0.0],
        "distance_from_airport": [3.0, 10.0, 25.0, 4.0, 8.0, 6.0],
        "number_of_amenities": [5.0, -1.2, 9.0, 7.0, 6.0, 8.0],
        "online_rating_score": [3.5, 4.0, 4.5, 3.0, 4.2, 3.8],
        "room_price_per_night": [150.0, 200.0, 300.0, 140.0, 210.0, 250.0],
    })


def test_output_shape():
    # 4 numeric + 3 one-hot (room_type) + 2 ordinal = 9 columns, one row per input row
    prep = DataPreparation(TEST_CONFIG)
    X = prep.fit_transform(prep.clean_data(_make_test_df()))
    assert X.shape == (6, 9)


def test_ordinal_mapping():
    # The two ordinal columns come last: season, then location_city_tier
    prep = DataPreparation(TEST_CONFIG)
    X = prep.fit_transform(prep.clean_data(_make_test_df()))
    assert list(X[:3, -2]) == [0, 1, 2]   # Low, Shoulder, High
    assert list(X[:3, -1]) == [2, 1, 0]   # Tier 1, Tier 2, Tier 3 (Tier 3 = 0)


def test_unseen_categories_do_not_crash():
    prep = DataPreparation(TEST_CONFIG)
    prep.fit(prep.clean_data(_make_test_df()))
    new = _make_test_df().head(2)
    new["room_type"] = "Penthouse"
    new["season"] = "Winter"
    X = prep.transform(prep.clean_data(new, training=False))
    assert X.shape == (2, 9) and np.isfinite(X).all()
    assert (X[:, -2] == -1).all()   # an unseen season is encoded as -1


def test_missing_values_are_imputed():
    prep = DataPreparation(TEST_CONFIG)
    prep.fit(prep.clean_data(_make_test_df()))
    new = _make_test_df().head(3).drop(columns="room_price_per_night")
    new.loc[0, "hotel_star_rating"] = np.nan   # numerical
    new.loc[1, "room_type"] = np.nan           # nominal
    new.loc[2, "season"] = np.nan              # ordinal
    X = prep.transform(prep.clean_data(new, training=False))
    assert not np.isnan(X).any()


def test_negative_amenities_clipped_and_input_unchanged():
    raw = _make_test_df()
    cleaned = DataPreparation(TEST_CONFIG).clean_data(raw)
    assert (cleaned["number_of_amenities"] >= 0).all()
    assert len(cleaned) == len(raw)                  # no rows removed
    assert (raw["number_of_amenities"] < 0).any()    # the caller's DataFrame is untouched


def test_schema_validation_errors():
    prep = DataPreparation(TEST_CONFIG)
    for bad_df, text in [(_make_test_df().drop(columns="season"), "missing required columns"),
                         (_make_test_df().iloc[0:0], "empty")]:
        try:
            prep.clean_data(bad_df)
        except ValueError as e:
            assert text in str(e)
        else:
            raise AssertionError(f"Expected a ValueError containing '{text}'")


def test_end_to_end_raw_rows_to_model_ready_matrix():
    prep = DataPreparation(TEST_CONFIG)
    # Training mode: raw rows (with target) -> clean -> fit_transform
    X_train = prep.fit_transform(prep.clean_data(_make_test_df()))
    assert X_train.shape == (6, 9) and np.isfinite(X_train).all()
    # Prediction mode: new raw rows (no target) -> clean -> transform with the fitted preprocessor
    new = _make_test_df().drop(columns="room_price_per_night").head(2)
    X_new = prep.transform(prep.clean_data(new, training=False))
    assert X_new.shape == (2, 9) and np.isfinite(X_new).all()


def run_tests() -> None:
    """Run every test_* function in this module and print the result (no pytest needed)."""
    tests = [f for name, f in globals().items() if name.startswith("test_") and callable(f)]
    for test in tests:
        test()
        print(f"PASSED  {test.__name__}")
    print(f"All {len(tests)} preprocessing tests passed.")


if __name__ == "__main__":
    run_tests()
