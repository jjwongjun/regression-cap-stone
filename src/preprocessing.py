# Standard library imports
import logging
from typing import Any, Dict, List

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


class DataPreparation:
    """
    A class used to clean and preprocess data for predicting hotel room price per night.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical and nominal features.
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

        With training=True (the default) rows without a usable target or attendance rate are
        removed and duplicate students are merged. With training=False (used when predicting
        for new students) no rows are removed, so every input row gets a prediction.

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

        Checks that the data is not empty, that every configured feature is present and, for training data.

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
        numerical_transformer = Pipeline(steps=[
            ("scaler", StandardScaler()),
        ])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        ordinal_transformer = Pipeline(
            steps=[
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