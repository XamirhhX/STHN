# Scale Observation Analysis: 20% Closer Two-Tile Demo

## Run Folder

```text
scaled_observation_examples_2026-05-18_08-02-26-20260518T080309Z-3-001/scaled_observation_examples_2026-05-18_08-02-26
```

Main files:

```text
scaled_observation_results.csv
drift_vs_missing_ratio.png
drift_vs_scale.png
clean_prediction.png
examples/
```

![Drift vs missing ratio](scaled_observation_examples_2026-05-18_08-02-26-20260518T080309Z-3-001/scaled_observation_examples_2026-05-18_08-02-26/drift_vs_missing_ratio.png)

## What Was Tested

This run simulated the professor's scale/altitude idea:

```text
20% closer to ground -> thermal footprint scale = 0.8
```

The script created smaller thermal observations and pasted them into a standard model input canvas.

Run configuration inferred from the CSV:

| Setting | Value |
|---|---:|
| Number of variations | 30 |
| Scale | 0.8 |
| Closer-percent convention | 20% |
| Number of pasted observations | 2 |
| Placement path | diagonal |
| Blend mode | average |
| Fill modes | zero, half, mean |
| Metric | prediction drift vs clean output |

Important: this is still an example-pair smoke test, not official dataset MACE/CE. It measures how much the model output changes relative to the clean prediction.

## Why The Missing Area Is Not 20%

The scale is `0.8`, so each individual thermal tile is 80% of the canvas width/height.

However, two such tiles are pasted into the same canvas. Because they cover different positions, the total visible coverage is much higher than one tile alone.

Observed coverage:

| Quantity | Value |
|---|---:|
| Mean coverage ratio | 90.16% |
| Mean no-data ratio | 9.84% |
| Minimum no-data ratio | 7.94% |
| Maximum no-data ratio | 13.04% |

So this run is best described as:

```text
20% scale reduction with two overlapping observations, producing about 8-13% no-data area.
```

It is not a 20% missing-area test.

## Overall Drift Statistics

| Statistic | Prediction drift |
|---|---:|
| Count | 30 |
| Mean | 42.32 |
| Median | 44.07 |
| Minimum | 11.87 |
| Maximum | 55.20 |
| P75 | 52.14 |
| P90 | 54.53 |
| P95 | 55.20 |

Interpretation:

- The model always changed relative to the clean prediction.
- There were no catastrophic jumps like the previous high-missing-area censorship test.
- Drift stayed in a moderate band, roughly `12-55` database-scale pixels.

## Threshold View

Using the same drift thresholds as before:

| Drift category | Count | Rate |
|---|---:|---:|
| Stable, `<= 10` | 0 / 30 | 0.0% |
| Moderate, `10-25` | 3 / 30 | 10.0% |
| Large, `25-50` | 14 / 30 | 46.7% |
| Failure, `> 50` | 13 / 30 | 43.3% |
| Catastrophic, `> 100` | 0 / 30 | 0.0% |

This means the 20% closer two-tile mosaic consistently affects the model, but does not create extreme instability in this run.

## Fill-Mode Comparison

| Fill mode | Count | Mean drift | Median drift | Min | Max | Failure `>50` |
|---|---:|---:|---:|---:|---:|---:|
| zero | 10 | 36.66 | 37.89 | 11.87 | 51.89 | 3 / 10, 30.0% |
| half | 10 | 44.26 | 42.66 | 29.18 | 55.20 | 4 / 10, 40.0% |
| mean | 10 | 46.04 | 51.69 | 24.00 | 54.45 | 6 / 10, 60.0% |

Interpretation:

- `zero` fill was best in this specific 20% closer run.
- `mean` fill had the highest failure rate under the `>50` drift threshold.
- The difference is not huge enough to claim a universal rule, because this is one example pair and only 30 variations.

## Missing-Ratio Bins

| No-data area | Count | Mean drift | Median drift | Failure `>50` |
|---|---:|---:|---:|---:|
| 7.5-9% | 9 | 53.49 | 54.45 | 9 / 9, 100.0% |
| 9-10.5% | 11 | 42.11 | 41.18 | 3 / 11, 27.3% |
| 10.5-12% | 7 | 38.73 | 39.72 | 1 / 7, 14.3% |
| 12-13.5% | 3 | 17.98 | 12.91 | 0 / 3, 0.0% |

This looks counterintuitive: lower missing area had larger drift.

The reason is likely placement geometry, not missing area alone. The deterministic low-missing case places the two 80% tiles at the exact diagonal corners:

```text
[(0, 0, 205, 205), (51, 51, 256, 256)]
```

This preserves high coverage but creates a strong duplicated/overlapped diagonal mosaic structure. Some jittered placements reduce coverage slightly but apparently create less harmful overlap geometry.

So the right interpretation is:

```text
For scaled mosaics, placement geometry matters as much as missing area.
```

## Largest Drift Cases

| Variation | Fill | Missing area | Coverage | Drift |
|---:|---|---:|---:|---:|
| 1 | half | 7.94% | 92.06% | 55.20 |
| 4 | half | 7.94% | 92.06% | 55.20 |
| 16 | half | 7.94% | 92.06% | 55.20 |
| 2 | mean | 7.94% | 92.06% | 54.45 |
| 29 | mean | 7.94% | 92.06% | 54.45 |
| 20 | mean | 8.17% | 91.83% | 53.01 |
| 10 | half | 9.60% | 90.40% | 52.45 |
| 23 | mean | 10.52% | 89.48% | 52.22 |
| 0 | zero | 7.94% | 92.06% | 51.89 |
| 14 | mean | 9.59% | 90.41% | 51.73 |

## Lowest Drift Cases

| Variation | Fill | Missing area | Coverage | Drift |
|---:|---|---:|---:|---:|
| 27 | zero | 13.04% | 86.96% | 11.87 |
| 21 | zero | 12.78% | 87.22% | 12.91 |
| 8 | mean | 10.76% | 89.24% | 24.00 |
| 7 | half | 12.79% | 87.21% | 29.18 |
| 12 | zero | 9.60% | 90.40% | 33.59 |

Again, this supports the idea that placement/overlap pattern is important.

## Main Conclusion

For the 20% closer case with two diagonal observations:

```text
The STHN prediction changes consistently, but not catastrophically.
```

The output drift is usually around `40-55` database-scale pixels. This suggests that the pretrained model is sensitive to scaled mosaic artifacts even when most of the canvas is covered.

## How To Explain This To The Professor

Use this wording:

> I changed the experiment from simple censorship to scaled partial observations. In this run I simulated the drone being 20% closer, so each thermal observation was scaled to 80% of the normal canvas. I pasted two such observations diagonally into one input canvas. Because there are two overlapping observations, the final no-data region was only about 8-13%, not 20%. The pretrained model's prediction drifted by a mean of 42.3 and a median of 44.1 database-scale pixels. There were no catastrophic failures above 100 pixels, but 43% of the variants exceeded a 50-pixel drift threshold. The main finding is that the model is sensitive not only to missing area, but also to the geometry of how scaled observations overlap.

## What This Run Supports

This run supports:

- The scale-mosaic generator works.
- The 20% closer convention produces scale `0.8`.
- Two scaled observations can be combined into one model input.
- The pretrained STHN model is affected by scaled mosaic structure.
- Placement/overlap geometry matters, not just total no-data area.

This run does not yet support:

- Official MACE/CE claims.
- Dataset-wide robustness claims.
- A final threshold for acceptable altitude/scale change.

## Recommended Next Run

The current run only tests fixed scale `0.8`.

To understand scale sensitivity, run a sweep:

```bash
python experiments/scaled_observation_examples_demo.py \
  --two_stages \
  --num_variations 100 \
  --min_scale 1.0 \
  --max_scale 0.6 \
  --num_tiles 2 \
  --path diagonal \
  --fill_modes zero half mean
```

This will answer:

```text
At what scale does the prediction start changing sharply?
```

Also test different paths:

```bash
--path horizontal
--path vertical
--path corners --num_tiles 4
```

The professor's idea is fundamentally about geometry, so path/placement should be tested explicitly.

