# Models, Features and Results

This page explains what the models are trained on, how they are chosen, what they achieve and
what to be careful about. For the code, see [api_reference.md](api_reference.md).

## The data

`data/synthetic_data_169.csv`: 1,000 rows and 8 columns, with no missing values and no duplicate
rows. The file looks synthetic (generated), so conclusions apply to this dataset and not
necessarily to real hotels.

| Column | Type | Notes |
|---|---|---|
| `room_price_per_night` | number (target) | Mean 209.85, standard deviation 66.83, range 85.66 to 523.82. Right-skewed (skew 1.35). |
| `hotel_star_rating` | number | **Already standardised** (mean about 0, standard deviation about 1, range -2.86 to 3.52). The negative values are valid; the column is not in real stars. |
| `online_rating_score` | number | Review score, range 6.19 to 9.96. |
| `number_of_amenities` | number | Count, mean 7.9. Eight rows are negative (-1.6 to -0.15), which is impossible, so they are set to 0. |
| `distance_from_airport` | number | Range 3.5 to 60.8; about 4% of rows are outliers (beyond 1.5 times the interquartile range). |
| `room_type` | text | Standard (600 rows), Deluxe (300), Suite (100). |
| `location_city_tier` | text | Tier 1 (500), Tier 2 (300), Tier 3 (200). |
| `season` | text | High (400), Shoulder (350), Low (250). |

## What the exploration showed

- Price rises with star rating (correlation 0.73), online rating (0.61) and number of amenities
  (0.57). These three are also correlated with each other (0.45 to 0.65).
- Price falls with distance from the airport (-0.25), a weak effect.
- Prices are higher in busier seasons (High about 233, Shoulder 203, Low 183) and in higher city
  tiers (Tier 1 about 237, Tier 2 196, Tier 3 163).
- **Room type runs against expectation:** Standard rooms are the most expensive on average
  (about 228), then Deluxe (187), then Suite (169). The pattern remains after controlling for
  the other features. Room type is therefore treated as an unordered category (one-hot encoded)
  and not as an ordered one.
- Season and city tier do have a natural order, so they are encoded as 0, 1, 2 in that order
  (Low < Shoulder < High; Tier 3 < Tier 2 < Tier 1, ordered by price).

## Preprocessing

Every model receives the same preprocessing, fitted on training data only:

- Numeric columns: missing values filled with the median, then `RobustScaler`. The robust scaler
  uses the median and the interquartile range, so the outliers in distance and similar columns
  do not distort the scale.
- `room_type`: missing values filled with the most frequent value, then one-hot encoded.
- `season` and `location_city_tier`: missing values filled with the most frequent value, then
  ordinal encoded.
- A value never seen in training does not crash the model: an unseen room type becomes all zeros
  and an unseen season or tier becomes -1.

The training data has no missing values; the imputation exists for new data.

## Engineered features

Three product features are added: `star_x_online`, `star_x_amenities` and `amenities_x_online`.

**Why.** Guests pay a premium for hotels that are good on several quality measures at once, so
the effect of one measure grows with the others. These three columns are the strongest
predictors. Only products are used because the star rating is standardised, which makes ratios
and logarithms of it meaningless.

**What they achieve.** Measured with 5-fold cross-validation repeated 3 times on the full data
(R²):

| Model | Without | With |
|---|---|---|
| Ridge | 0.728 | **0.779** |
| Random forest | 0.736 | 0.736 |
| HistGradientBoosting | 0.774 | 0.772 |

Linear models gain about 0.05. Tree models are unchanged, because they find interactions on
their own.

**What was tried and dropped:** squares of the numeric columns (same gain as the interactions,
not additive), season × city tier, room type × star rating, a composite quality score, log or
squared distance, and several flags. None added more than noise on top of the interactions.

## Feature selection

`rank_features` scores every model-ready feature with four methods (F-test, mutual information,
Lasso, random forest importance) on the training split. The top of the table from the latest run:

| Feature | Average rank | Comment |
|---|---|---|
| `star_x_online` | 1.75 | Engineered |
| `star_x_amenities` | 2.00 | Engineered |
| `hotel_star_rating` | 2.50 | |
| `online_rating_score` | 4.00 | |
| `amenities_x_online` | 6.25 | Engineered |
| `season` | 6.75 | |
| `location_city_tier` | 7.00 | |
| `number_of_amenities` | 8.13 | Its information overlaps with star rating (correlation 0.65), so Lasso gives it zero weight |
| `room_type_Standard` | 8.25 | |
| `distance_from_airport` | 9.25 | Weak alone, but the forest does use it |
| `room_type_Deluxe`, `room_type_Suite` | 10.88, 11.25 | |

The full table is saved as `feature_ranking.csv` in every run folder.

**Removing features did not help.** Keeping only the best k features (`n_selected_features`)
lowered accuracy: keeping 10 of 12 was neutral to slightly worse, 8 lost about 0.06 R² and 6 lost
more than 0.1. Features that look weak on their own (distance, season, room type) still add
information in combination. The default is therefore to keep all 12 and use the ranking to
understand the data, not to prune it.

## Models

Ten models are compared. Six are baselines with default settings; four are tuned.

| Model | Type | Why it is included |
|---|---|---|
| `dummy_mean` | Predicts the mean price | The benchmark every real model must beat |
| `linear_regression` | Linear | Simple, interpretable reference |
| `ridge`, `ridge_tuned` | Linear with L2 penalty | Stabilises coefficients when features are correlated, as here |
| `lasso`, `lasso_tuned` | Linear with L1 penalty | Can set weak features to exactly zero |
| `random_forest`, `rf_tuned` | Many decision trees averaged | Captures non-linear effects and interactions; robust to outliers |
| `hist_gradient_boosting`, `hgb_tuned` | Trees built one after another, each fixing the previous errors | Usually the strongest on tabular data |

Tuned models use a grid search with 5-fold cross-validation on the training set (grids in
[configuration.md](configuration.md)).

## How the final model is chosen

1. Split the data 80/10/10 (800 train, 100 validation, 100 test) with a fixed seed.
2. Train every model on the training set and score it on the validation set.
3. Pick the model with the highest validation R².
4. Refit that model on train + validation (900 rows), keeping its parameters.
5. Evaluate it once on the test set. The test set is never used for any decision.

## Results

Validation set (100 rows):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| `dummy_mean` | 49.11 | 65.64 | -0.002 |
| `linear_regression` | 21.71 | 28.13 | 0.816 |
| `ridge` | 22.23 | 28.81 | 0.807 |
| `lasso` | 22.47 | 29.55 | 0.797 |
| `random_forest` | 21.46 | 29.87 | 0.793 |
| `hist_gradient_boosting` | 22.38 | 30.71 | 0.781 |
| `ridge_tuned` | 21.87 | 28.29 | 0.814 |
| `lasso_tuned` | 21.78 | 28.20 | 0.815 |
| `rf_tuned` | 21.29 | 30.02 | 0.790 |
| **`hgb_tuned`** | **20.04** | **27.33** | **0.826** |

Cross-validated R² on the training set (average over 5 folds) and the parameters chosen:

| Model | CV R² | Best parameters |
|---|---|---|
| `ridge_tuned` | 0.776 | alpha 0.1 |
| `lasso_tuned` | 0.776 | alpha 0.01 |
| `hgb_tuned` | 0.771 | learning rate 0.05, depth 2, 600 iterations, minimum 40 samples per leaf |
| `rf_tuned` | 0.730 | no depth limit, all features per split, minimum 3 samples per leaf |

Selected model `hgb_tuned`, refitted on train + validation and evaluated on the test set:

| MAE | RMSE | R² |
|---|---|---|
| 22.45 | 30.71 | 0.661 |

In plain terms, a typical prediction is off by about 22 per night on prices that average 210.
Every real model explains far more than the mean-only baseline.

These results are the same with the pinned library versions in `requirements.txt` and with newer
ones (scikit-learn 1.8).

## How far to trust the numbers

- **The validation and test sets have only 100 rows each.** Differences of a few hundredths in R²
  between models (0.79 to 0.83) are within noise. Do not read the ranking among the top models
  as settled.
- **The models are closer than the selection suggests.** Cross-validation, which uses all 800
  training rows, puts the two linear models marginally ahead of the selected boosting model
  (0.776 against 0.771), and even plain linear regression scored 0.816 on validation. Boosting
  was chosen because it had the best single validation score.
- **The test R² (0.661) is lower than the validation R² (0.826) partly by chance of the split.**
  The test rows are less spread out (standard deviation of price 53, against 66 overall), and the
  same absolute error gives a lower R² when the spread is smaller. The error in price units
  (MAE 22.5 and RMSE 30.7, against 20.0 and 27.3 on validation) differs far less than the R²
  does. Quote the MAE and RMSE as well as R².
- **The data looks synthetic.** Real hotel prices depend on factors that are not in this dataset.
- **`hotel_star_rating` is standardised.** New data must be standardised in the same way (with the
  same mean and standard deviation as the training data, which are not stored). See
  [deployment.md](deployment.md).
- **The model should not be used outside the training range** (for example prices above about
  520 or far-from-airport distances beyond 61); tree models cannot extrapolate.

## Possible improvements

- Choose the final model by cross-validation score instead of one 100-row validation set, or
  compare models with repeated cross-validation.
- Model the logarithm of price. In testing it raised the linear models from 0.727 to 0.785 R²,
  because price is right-skewed and its effects look multiplicative. It needs a small change in
  `model_optimization.py` and has not been added.
- Report prediction intervals, not only single values.
- Test on real data before any real use.
