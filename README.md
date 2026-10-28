# Literature Review: Machine Learning for Regression

## 1. Best Practices

- **Start with a simple baseline** (mean predictor, linear/ridge regression) so every complex model has a benchmark to beat.
- **Split data correctly**: use train/validation/test sets; use k-fold cross-validation (typically 5 or 10) for model selection; use time-based splits for temporal data and group-based splits when rows are related.
- **Prevent data leakage**: fit all preprocessing (scaling, imputation, encoding) on training folds only, ideally inside a pipeline.
- **Inspect the target**: check skew and outliers; consider a log or Box-Cox transform for right-skewed targets (e.g., prices).
- **Tune hyperparameters systematically**: random search or Bayesian optimization (e.g., Optuna) beats manual tuning and grid search at the same budget (Bergstra & Bengio, 2012).
- **Check residuals**: plot residuals vs. predictions to detect non-linearity, heteroscedasticity, and bias.
- **Ensemble at the end**: averaging or stacking diverse models usually gives a small but reliable gain.
- **Keep it reproducible**: fix random seeds, version data and code, and log experiments.

## 2. Methodologies

- **Classical statistical**: Ordinary Least Squares, with Ridge (L2), Lasso (L1), and Elastic Net regularization to handle multicollinearity and overfitting (Hastie et al., *The Elements of Statistical Learning*).
- **Non-linear extensions**: polynomial regression, GAMs, and Support Vector Regression for smooth non-linear relationships on small to medium data.
- **Tree-based learning**: decision trees, bagging (Random Forest; Breiman, 2001), and boosting (Gradient Boosting; Friedman, 2001).
- **Neural approaches**: MLPs and specialized tabular networks, which generally need more data and tuning.
- **Probabilistic methods**: Gaussian Processes, quantile regression, and conformal prediction when uncertainty estimates or prediction intervals are needed.
- **Typical workflow**: define problem and metric → EDA → baseline → feature engineering → model comparison with CV → tuning → final test evaluation → error analysis.

## 3. Model Architectures

| Model family | Strengths | Limitations |
|---|---|---|
| Linear / Ridge / Lasso | Fast, interpretable, strong baseline | Cannot capture complex interactions |
| Random Forest | Robust, little tuning, handles non-linearity | Large models, cannot extrapolate |
| Gradient Boosting (XGBoost, LightGBM, CatBoost) | Top accuracy on most tabular data; CatBoost handles categoricals natively | Needs tuning; can overfit noisy data |
| Neural networks (MLP, ResNet, FT-Transformer) | Flexible, good for very large or multimodal data | Often underperform trees on typical tabular data |
| TabPFN (tabular foundation model) | Pretrained transformer, no per-dataset training; reported to beat tuned boosting on small data (up to ~10,000 rows; Hollmann et al., *Nature*, 2025) | Limited to small/medium data; newer, less proven in production |

**Key findings from the literature:**
- Tree-based models still outperform deep learning on typical medium-sized tabular data (Grinsztajn et al., 2022).
- A large benchmark of 19 algorithms on 176 datasets (TabZilla; McElfresh et al., 2023) found no single winner. Boosted trees tend to win on larger, more irregular datasets, while the NN-vs-GBDT gap often matters less than proper tuning.
- Tabular foundation models such as TabPFN are the main recent challenger on small datasets, and recent domain studies (e.g., solar energy regression, 2026) report gains over gradient boosting. Independent validation is still ongoing.
- **Practical recommendation:** use gradient boosting as the main candidate, linear models as the baseline, and TabPFN as an additional candidate if data is small.

## 4. Feature Engineering Techniques

- **Missing values**: median/mean imputation, indicator flags for missingness, or model-based imputation (KNN, iterative). Tree libraries like XGBoost and LightGBM handle missing values natively.
- **Categorical encoding**: one-hot for low cardinality; target/ordinal encoding (with cross-validated fitting to avoid leakage) for high cardinality; CatBoost's ordered target statistics as a built-in option.
- **Scaling**: standardization or min-max for linear, SVM, and neural models; unnecessary for tree-based models.
- **Transformations**: log, square-root, or power transforms for skewed features; binning for non-linear effects.
- **Interaction and polynomial features**: useful for linear models; trees learn interactions on their own.
- **Domain features**: ratios, differences, aggregations, and date/time components (hour, weekday, season, lags, rolling statistics) often give the largest gains.
- **Outlier handling**: winsorizing or robust scalers; keep outliers if they are genuine signal.
- **Feature selection**: remove near-constant and highly correlated features; use L1 regularization, permutation importance, or SHAP values to rank and prune.
- **Dimensionality reduction**: PCA when features are numerous and correlated.

## 5. Evaluation Metrics

| Metric | Formula / idea | When to use |
|---|---|---|
| **MAE** | Mean of \|y − ŷ\| | Robust to outliers; easy to interpret in target units |
| **MSE / RMSE** | Mean of (y − ŷ)², and its square root | Penalizes large errors; RMSE is in target units |
| **R²** | 1 − SS_res / SS_tot | Share of variance explained; compare against baseline |
| **Adjusted R²** | R² penalized for number of predictors | Comparing models with different feature counts |
| **MAPE / sMAPE** | Mean percentage error | Relative error; avoid when targets are near zero |
| **RMSLE** | RMSE on log(1 + y) | Skewed targets where relative error matters |
| **Median AE / Huber loss** | Median or hybrid error | Heavy-tailed errors and outliers |
| **Pinball loss / interval coverage** | Quantile and interval accuracy | Probabilistic predictions and uncertainty |

**Guidance:**
- Choose the metric to match the business cost of errors, and report at least two (e.g., RMSE and MAE) plus R².
- Always report metrics on a held-out test set and, where possible, mean ± standard deviation across CV folds.
- Compare against the baseline model, and complement metrics with residual plots and error analysis by segment.
- Avoid relying on R² alone: it can look high for poor models on trending data and is not comparable across different targets.

# Room Price Per Night Prediction

## Project Description
This project implements a machine learning pipeline to predict the hotel room price per night.
The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple regression models are trained and compared against a trivial baseline to find the best performing one based on various metrics.

## Prerequisites and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.11
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:

git clone https://github.com/jjwongjun/regression-cap-stone.git
cd regression-cap-stone

2. Create and activate a conda environment:
    ```
    conda create -n hdbenv python=3.11
    conda activate hdbenv
    ```

3. Install the required packages:

pip install -r requirements.txt

## Pipeline Execution
To run the complete pipeline, execute:

python main.py

A different configuration file can be given with `python main.py --config path/to/config.yaml` (default: `./src/config.yaml`).

This will:
1. Load and validate the configuration, then load the data (shape and missing values are logged)
2. Validate the data schema and clean the data (the number of rows removed at each step is logged)
3. Preprocess features
4. Split data into training, validation, and test sets (the target distribution of every split is logged)
5. Train baseline models (a mean-prediction dummy plus five real models)
6. Perform hyperparameter tuning (best parameters are logged and the full cross-validation results saved)
7. Evaluate models and select the best one by validation R²
8. Refit the selected model on the training and validation data combined, and evaluate it once on the test set
9. Save the model, metrics, test predictions, tuning results, config snapshot and diagnostic plots to a timestamped folder in `artifacts/`, and write a log to `logs/pipeline.log`

The two output locations are set by `artifacts_dir` and `log_file` in the config. Remove either key to switch it off. The versions of Python and the main libraries and the random seed are logged at start-up.

### Predicting for new hotel rooms
A saved model can score new hotels without retraining:

python main.py --predict new_hotels.csv --model artifacts/run_<timestamp>/best_model.joblib --output predictions.csv

The input CSV needs the same columns as the training data except `room_price_per_night`. In this mode no rows are removed, so the output has exactly one prediction per input row.

### Checking the helper functions
The cleaning and sanity-check helpers contain doctests that act as unit tests. Run them from the project root with:

python -m doctest -v src/data_preparation.py src/model_training.py

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Random seed (`random_seed`), used for every split and model
- Train/validation/test split ratios (`val_test_size` is the held-out share, `val_size` is the share of that which becomes the test set; 0.2 and 0.5 give 80/10/10)
- Hyperparameter grid for model tuning, cross-validation folds (`cv`) and the tuning metric (`scoring`)
- Output locations for artifacts and logs

### 2. Data Preparation (`DataPreparation` class)
The DataPreparation' class handles:
- Checking that the data is not empty and all required columns exist (clear error message otherwise)
- Convert negative `number_of_amenities` values to 0
- Sanity checks on the cleaned data: expected features are present
- Creating a preprocessor for feature transformation (a copy of it is used for every model, so fitted state never leaks between models)

### 3. Model Training (`ModelTraining` class)
The ModelTraining' class manages:
- Splitting data into training, validation, and test sets (with split-size checks and target distributions logged)
- Training a mean-prediction baseline and baseline models (Linear Regression, Ridge, Lasso, Random Forest, Histogram Gradient Boosting)
- Hyperparameter tuning for Ridge, Lasso, Random Forest, Histogram Gradient Boosting models
- Evaluating models using multiple metrics (MAE, MSE, RMSE, R²) with predicted-vs-actual and residual plots
- Selecting the best model based on R² score on the validation set
- Refitting the selected model on training + validation data and evaluating it once on the test set
- Saving the model, metrics, test predictions, tuning results and config snapshot

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading configuration and data
- Initializing data preparation and model training
- Running the training and evaluation process
- Identifying and evaluating the best model on the test set

## Key Findings and Feature Handling

### Exploratory Data Analysis

Before diving into the model training process, an exploratory data analysis (EDA) was conducted to understand the data better and identify any potential issues. Here are some key findings from the EDA:

**EDA Findings and Explanations**: 
- The data has 1000 rows and 8 columns (5 numeric, 3 text)
- No missing values
- Negative values for `hotel_star_rating` and `number_of_amenities`
- No duplicates
- `hotel_star_rating` appears pre-standardized with z-score transformation
- 8 negative `number_of_amenities` rows, with values between −1.6 and −0.15. A count of amenities can’t be negative, so most likely noise in a column whose true value is low, near 0. Prices back this up:
- Price decreases with higher Tier number
- Price decreases with Standard > Deluxe > Suite
- Price increases with busier season
- Positive trend as Hotel Star Rating increases, room price per night increases
- Positive trend as online rating score increases, room price per night increases
- No clear trend between Distance from Airport vs room price per night
- Positive trend as number of amentities increases, room price per night increases
- All three categorical feature (location_city_tier, room_type, season) are highly significant, so each carries real signal
- Strong positive correlation between 'hotel_star_rating' and 'room_price_per_night'. Higher hotel star rating tend to have higher room price per night.
- Strong positive correlation between 'online_rating_score' and 'room_price_per_night'. Higher online rating score tend to have higher room price per night.
- Strong positive correlation between 'number_of_amenities' and 'room_price_per_night'. Higher number of amenities tend to have higher room price per night.
- Strong positive monotonic correlation between 'hotel_star_rating' and 'room_price_per_night'. Higher hotel star rating score tend to have higher room price per night.
- Strong positive monotonic correlation between 'online_rating_score' and 'room_price_per_night'. Higher online rating score tend to have higher room price per night.
- Strong positive monotonic correlation between 'number_of_amenities' and 'room_price_per_night'. Higher number of amenities tend to have higher room price per night

### Data Cleaning
- Convert negative `number_of_amenities` values to 0

### Feature Processing
- **Numerical Features**: `hotel_star_rating`, `distance_from_airport`, `number_of_amenities`, `online_rating_score`
- **Nominal Features**: `room_type` - Encoded using `OneHotEncoder` because even though there is intrinsic order, order is inverse vs. typical expectations
- **Ordinal Features**: `season`, `location_city_tier` - Encoded using `OrdinalEncoder` because there is intrinsic order in these features
- Any column not listed in the configuration is dropped by the preprocessor, so an unexpected column can never reach a model silently

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Linear Regression**: Basic model without regularization; it sets the linear baseline for comparison with other models
2. **Ridge Regression**: Linear regression with L2 regularization because it includes a regularization term to prevent overfitting. Since different input features will have varying levels of influence on the room price, the reduction of coefficient size caused by the shrinkage penalty translates to each predictor feature having less influence on the final prediction
3. **Lasso Regression**: Linear regression with L1 regularization because it includes a regularization term to prevent overfitting. It is similar to Ridge Regression except it not only shrinks the coefficients towards 0 but can also force some coefficients to be exactly 0, effectively performing built-in feature selection by excluding irrelevant features from the model
4. **Random Forest**: Ensemble of decision trees; captures non-linear effects and interactions between features without manual feature engineering
5. **Histogram Gradient Boosting**: Boosted trees that are usually strongest on tabular data

### Hyperparameter Tuning
- Grid search is performed for Ridge, Lasso, Random Forest, Histogram Gradient Boosting models because tuning model parameters can achieve the best possible performance. Grid Search CV's process is exhaustive but it ensures the best combination
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance
- A failed fit stops the run instead of being scored silently as missing. The best parameters and cross-validation scores are logged, saved to `metrics.json`, and the full cross-validation results are saved per model
- The final model is refitted on training + validation data before the one-time test evaluation, so it uses all the data available once model selection is finished

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual value
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual value
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same unit as the target
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model
- **Diagnostic plots**: Predicted vs actual, residuals vs predicted, and the residual distribution, to check for bias and for score ranges where the model does worse

## Deployment Considerations
- Code is structured into modular well-organized components making code easier to maintain, more scalable, and ready for real world applications
- The configuration is validated at start-up, the data schema is checked before cleaning, and the cleaned data, split sizes and preprocessed features are sanity-checked before training
- Console and file logging with timestamps; file, config and data errors are reported with clear messages and a non-zero exit code
- The selected model (`best_model.joblib`), its test metrics and predictions, tuning results and a snapshot of the configuration are saved to a timestamped folder in `artifacts/`, so a result can be reproduced and runs can be compared
- A saved model can be used on new hotel entries with `--predict`, without retraining

### Known Limitations and Future Work
- There is no separate `tests/` folder; the helper functions and sanity checks are covered by doctests only. Adding `pytest` unit tests and an end-to-end smoke test is the next step