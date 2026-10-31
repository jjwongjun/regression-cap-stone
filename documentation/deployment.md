# Deployment Guide

How to install the project, train and release a model, score new rooms, and keep the model
healthy. For the settings see [configuration.md](configuration.md); for the code see
[api_reference.md](api_reference.md).

## 1. Install

Requirements: Python 3.11 (the project was also run on 3.12.3) and the packages in
`requirements.txt`, which are pinned:

| Package | Version |
|---|---|
| scikit-learn | 1.4.2 |
| pandas | 2.2.1 |
| numpy | 1.26.4 |
| joblib | 1.4.2 |
| PyYAML | 6.0.1 |
| matplotlib | 3.8.4 |
| seaborn | 0.13.2 |

```
conda create -n hotelenv python=3.11
conda activate hotelenv
pip install -r requirements.txt
python main.py --test
```

The last command should print "All 10 preprocessing tests passed" and "All 4 feature engineering
tests passed". The full pipeline was run with these exact versions (and separately with newer
ones) and gave the same results.

## 2. Train a model

```
python main.py
```

It took about 3 minutes in testing (most of it is the grid searches) and writes
a new `artifacts/run_<timestamp>/` folder (files listed in
[api_reference.md](api_reference.md#files-written-by-a-run)) and appends to `logs/pipeline.log`.

**Running on a server or in a scheduled job.** The code opens each diagnostic plot with
`plt.show()`. On a machine with a display that pauses the run until you close each window. Use a
non-interactive backend so it runs unattended (the plots are still saved):

```
MPLBACKEND=Agg python main.py                 # Linux and macOS
$env:MPLBACKEND = "Agg"; python main.py       # Windows PowerShell
```

The exit status is 0 on success and 1 on failure, so a scheduler or script can check it.

## 3. Score new rooms

```
python main.py --predict new_rooms.csv \
    --model artifacts/run_<timestamp>/best_model.joblib \
    --output predictions.csv
```

### Input file

A CSV with one row per room and these columns (any other columns are ignored; the price column is
not needed):

| Column | Allowed values and notes |
|---|---|
| `room_type` | `Standard`, `Deluxe` or `Suite` |
| `location_city_tier` | `Tier 1`, `Tier 2` or `Tier 3` |
| `season` | `Low`, `Shoulder` or `High` |
| `hotel_star_rating` | A **standardised** number (a z-score: 0 is an average hotel, 1 is one standard deviation above). It is not a number of stars. Training values ranged from -2.9 to 3.5. |
| `distance_from_airport` | A number in the same unit as the training data (training range 3.5 to 60.8) |
| `number_of_amenities` | A count (training range 0 to 17.6); negative values are set to 0 |
| `online_rating_score` | A score on the training scale (6.2 to 10.0) |

Example `new_rooms.csv`:

```
room_type,location_city_tier,season,hotel_star_rating,distance_from_airport,number_of_amenities,online_rating_score
Deluxe,Tier 1,High,0.8,6.5,10.2,9.3
Standard,Tier 3,Low,-0.6,22.0,5.1,8.4
Suite,Tier 2,Shoulder,1.5,12.0,13.0,9.6
```

### Output file

One row per input row, in the same order:

```
row,predicted_room_price_per_night
0,308.40
1,141.83
2,257.13
```

`row` is the position of the row in the input file. To match predictions to your own room ids,
join on position, or keep your id column next to the prediction in your own script.

### How imperfect input is handled

| Situation | What happens |
|---|---|
| A number is blank | Filled with the training median |
| A category is blank | Filled with the most frequent training category |
| A category was never seen in training (for example `Penthouse`) | The room is still scored: an unseen room type is treated as none of the known types, and an unseen season or tier is encoded as -1 (below the lowest known level). Treat such predictions with caution. |
| A required column is missing | The run stops with a message naming the column |
| `number_of_amenities` is negative | Set to 0 |

### The star rating needs the same scale

`hotel_star_rating` in the training data was already standardised, and the mean and standard
deviation used are not stored anywhere. Real star counts (1 to 5) given to the model would be
read as huge z-scores and produce nonsense. Before using the model on data from another source,
find out how the training column was produced and apply the same transformation.

## 4. Release and version models

- Every run creates its own timestamped folder, so earlier models are never overwritten. A
  folder holds the model, its test metrics, the grid-search results and a copy of the
  configuration (`config_snapshot.yaml`), which is enough to reproduce or audit it.
- The log records the Python, pandas, numpy and scikit-learn versions and the seed. A saved model
  is a pickled scikit-learn pipeline: load it with the **same scikit-learn version** it was
  trained with (the pinned 1.4.2), or scikit-learn may warn or fail.
- Only load `.joblib` files you created or trust. Loading one can run code.
- To release a model, copy its run folder to wherever the scoring job reads from, and record the
  folder name with every set of predictions.

## 5. Retrain

Retrain when new priced data arrives, or when monitoring (next section) shows the model is
drifting:

1. Add the new rows to the training CSV (same columns as `data/synthetic_data_169.csv`) or point
   `file_path` at the new file.
2. Run `python main.py --test`, then `python main.py`.
3. Compare the new run's `metrics.json` and `cv_results_*.csv` with the current release. With a
   test set of 100 rows, also compare MAE and RMSE and do not rely on small R² differences.
4. Release the new folder only if it is at least as good.

If you change features or settings, update `src/config.yaml` and see the notes in
[configuration.md](configuration.md#features).

## 6. Monitor

Once the model is used on real traffic, watch:

- **Inputs.** Count blank values, unseen categories and values outside the training ranges
  (table in section 3). Many of these mean the data source changed.
- **Predictions.** The training prices ranged from about 86 to 524. Predictions far outside that,
  or a sudden shift in their average, are a warning sign.
- **Error.** When actual prices become known, compute the MAE and compare it with the test MAE
  of about 22.5. A steady rise means it is time to retrain.
- **Runs.** Check the exit status and the log after every scheduled training run.

## 7. Production readiness

| Area | Status |
|---|---|
| Code organisation | Separate modules for entry point, preprocessing, feature engineering and training; all settings in one YAML file |
| Reproducibility | Pinned requirements, one seed, saved configuration copy per run |
| Input checks | Config, data schema and preprocessed data are validated, with clear error messages |
| Logging and exit codes | Timestamped console and file log; exit status 0 or 1 |
| Tests | 14 small tests (`python main.py --test`) plus doctests (`python -m doctest -v src/preprocessing.py src/feature_engineering.py src/model_optimization.py`) |
| Not yet in place | A separate `tests/` folder with pytest tests for `model_optimization.py` and `main.py`; an automatic test run on every change (continuous integration); stored scaling values for `hotel_star_rating`; an id column in the prediction output; prediction intervals |

## 8. Troubleshooting

| Message or symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src.model_training'` | An old `main.py`. The module is now `src/model_optimization.py`. |
| `Config is missing required keys` | See [configuration.md](configuration.md). |
| `Data file not found` | Check `file_path` and run from the project root. |
| `Input data is missing required columns: [...]` | The CSV lacks one of the seven feature columns (plus the target for training). |
| `max_depth ... Got 'None' instead` | Write `null`, not `None`, in the YAML. |
| The run seems stuck on a plot window | Set `MPLBACKEND=Agg` (section 2). |
| A warning or error when loading a model | The scikit-learn version differs from the one used for training. |
| Predictions look implausible | Check the `hotel_star_rating` scale and the units of the other columns (section 3). |
