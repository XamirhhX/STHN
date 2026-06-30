# H5 Guide For Scaled Partial-Observation Evaluation

## 1. Difference From The Previous Censoring Test

The old test removed blocks from an otherwise normal thermal image.

The new scale test creates a new input image:

```text
multiple smaller thermal observations
placed into one standard canvas
with no-data regions where no observation exists
```

This represents the idea that the UAV is closer to the ground and each thermal observation covers a smaller footprint.

## 2. Minimal H5 Dataset Layout

For real MACE/CE results, use:

```text
STHN_DATASETS/minimal/
  maps/satellite/20201117_BingSatellite.png
  satellite_0_thermalmapping_135_train/test_queries.h5
  satellite_0_thermalmapping_135_train/test_database.h5
```

## 3. Inspect The H5 Files

```bash
python scripts/inspect_sthn_h5.py \
  --h5 /content/drive/MyDrive/STHN_DATASETS/minimal/satellite_0_thermalmapping_135_train/test_queries.h5
```

```bash
python scripts/inspect_sthn_h5.py \
  --h5 /content/drive/MyDrive/STHN_DATASETS/minimal/satellite_0_thermalmapping_135_train/test_database.h5
```

You should see keys like:

```text
image_name
image_data
```

The database file may only need `image_name` because satellite crops are extracted from the large map PNG.

## 4. Run The Scaled H5 Evaluation

For the professor's 20% closer case:

```bash
python experiments/scaled_observation_eval.py \
  --datasets_folder /content/drive/MyDrive/STHN_DATASETS/minimal \
  --dataset_name satellite_0_thermalmapping_135_train \
  --split test \
  --two_stages \
  --batch_size 1 \
  --num_variations 100 \
  --closer_percent 20 \
  --num_tiles 2 \
  --path diagonal \
  --fill_modes zero half mean
```

Output:

```text
outputs/scaled_observation_h5_<timestamp>/
  scaled_observation_h5_results.csv
  mace_vs_missing_ratio.png
  mace_vs_scale.png
  examples/
```

## 5. Meaning Of Output Columns

```text
scale
```

Linear scale of each thermal observation inside the standard canvas. `0.8` means 20% smaller.

```text
closer_percent_convention
```

`(1 - scale) * 100`. For scale `0.8`, this is `20`.

```text
coverage_ratio
```

Fraction of the final canvas covered by at least one pasted observation.

```text
missing_ratio
```

Fraction of the final canvas with no observation.

```text
num_tiles
```

How many smaller observations were pasted into the canvas.

```text
placements_xyxy
```

Pixel coordinates of each pasted observation in the final thermal canvas.

```text
mace_database
```

True MACE at database/satellite scale. This is the main error metric when using `.h5`.

```text
ce_database
```

True center error at database/satellite scale.

## 6. Scale Sweep

Instead of a fixed 20% closer condition, test a range:

```bash
python experiments/scaled_observation_eval.py \
  --datasets_folder /content/drive/MyDrive/STHN_DATASETS/minimal \
  --dataset_name satellite_0_thermalmapping_135_train \
  --split test \
  --two_stages \
  --batch_size 1 \
  --num_variations 100 \
  --min_scale 1.0 \
  --max_scale 0.6 \
  --num_tiles 2 \
  --path diagonal \
  --fill_modes zero half mean
```

This tests from normal footprint to 40% smaller footprint.

## 7. Recommended First H5 Experiment

Use:

```text
scale = 0.8
num_tiles = 2
path = diagonal
fill_modes = zero half mean
```

Reason:

- It matches the professor's 20% closer example.
- Two observations create realistic uncovered/no-data regions.
- Diagonal placement approximates motion with both x and y displacement.
- Multiple fill modes show whether no-data handling matters.

## 8. Recommended Second H5 Experiment

Run a scale sweep:

```text
scale 1.0 -> 0.6
```

This answers:

```text
At what scale / missing-ratio does the homography error start increasing sharply?
```

## 9. Important Interpretation

If you use the example demo:

```text
metric = prediction drift vs clean prediction
```

If you use `.h5`:

```text
metric = true MACE / true center error
```

For professor-facing results, prefer the `.h5` output when available.

