# Missing-Data Robustness Smoke Test Notes

## One-Sentence Summary

I deployed the STHN missing-data experiment in Google Colab and verified that the pretrained model, masking pipeline, CSV logging, and plotting work. Because the official STHN dataset was unavailable through Hugging Face, this run uses the repository's bundled example image pair and measures prediction drift against the clean prediction, not true ground-truth homography error.

## What Was Tested

The goal was to simulate incomplete/censored test-time input and observe how the pretrained STHN model responds without retraining.

Input pair:

- Satellite image: `examples/img1.png`
- Thermal image: `examples/img2.png`
- Model: pretrained STHN two-stage model
- Censored input: thermal image
- Number of variations: 30
- Censored area range: about 10% to 70%
- Mask types: 1 or 2 rectangular masks
- Fill modes: `zero`, `half`, `interp`

Output files:

- CSV: `missing_data_examples_2026-05-06_02-38-36-20260506T023925Z-3-001/missing_data_examples_2026-05-06_02-38-36/missing_data_examples_results.csv`
- Plot: `missing_data_examples_2026-05-06_02-38-36-20260506T023925Z-3-001/missing_data_examples_2026-05-06_02-38-36/prediction_drift_vs_censored_ratio.png`
- Clean prediction visualization: `missing_data_examples_2026-05-06_02-38-36-20260506T023925Z-3-001/missing_data_examples_2026-05-06_02-38-36/clean_prediction.png`

![Prediction drift vs censored ratio](missing_data_examples_2026-05-06_02-38-36-20260506T023925Z-3-001/missing_data_examples_2026-05-06_02-38-36/prediction_drift_vs_censored_ratio.png)

## Important Limitation

This is not the final professor-facing quantitative result yet. The official dataset contains numeric ground-truth homography/flow targets, but the bundled examples only provide images. Therefore:

- Final dataset experiment metric: true MACE / center error against ground truth.
- This smoke-test metric: prediction drift relative to the clean model prediction.

Interpretation:

If drift is small, the model's prediction is stable under that mask.
If drift is large, the mask caused a major prediction change or failure.

This still validates the experimental workflow and gives useful qualitative evidence, but it should be presented as a deployment and robustness-probing test, not as final benchmark accuracy.

## Clean Model Prediction

For the unmasked example pair, the model predicted the following four-corner displacement at `256x256` scale:

| Corner | dx | dy |
|---|---:|---:|
| Top-left | 104.73 | 10.27 |
| Top-right | -68.77 | 10.27 |
| Bottom-left | 104.69 | -163.35 |
| Bottom-right | -68.78 | -163.32 |

Scaled to the `1536x1536` satellite scale:

| Corner | dx | dy |
|---|---:|---:|
| Top-left | 628.36 | 61.64 |
| Top-right | -412.62 | 61.64 |
| Bottom-left | 628.13 | -980.08 |
| Bottom-right | -412.71 | -979.90 |

This clean prediction is used as the pseudo-reference for the smoke test.

## Main Results

Overall prediction drift statistics across 30 masked variations:

| Statistic | Drift vs Clean |
|---|---:|
| Count | 30 |
| Mean | 79.37 |
| Median | 17.24 |
| Minimum | 0.57 |
| Maximum | 411.60 |
| P90 | 370.93 |
| P95 | 374.82 |

Interpretation:

- The median drift is modest, so many masks do not completely destabilize the model.
- The mean is much larger than the median because a few severe failures dominate the average.
- The high P90/P95 values show a heavy-tail failure pattern: most cases are manageable, but some masks produce large prediction jumps.

## Before vs After Censorship

For this smoke test, the clean unmasked prediction is the reference. Therefore the "before censorship" drift is exactly `0` by definition.

| Condition | Censored area | Mean drift | Median drift | Stable rate | Failure rate |
|---|---:|---:|---:|---:|---:|
| Before censorship | 0.0% | 0.00 | 0.00 | 100.0% | 0.0% |
| After censorship, all masks | 36.8% average | 79.37 | 17.24 | 36.7% | 20.0% |

Definitions used here:

- Stable: drift `<= 10` database-scale pixels.
- Moderate drift: `10 < drift <= 25`.
- Large drift: `25 < drift <= 50`.
- Failure: drift `> 50`.
- Catastrophic failure: drift `> 250`.

These thresholds are heuristic for this smoke test. They are useful for presentation because they convert the plot into rates, but the final dataset run should use true MACE/CE thresholds chosen by the lab or professor.

## Stability And Failure Rates

Across all 30 censored variations:

| Drift category | Count | Rate |
|---|---:|---:|
| Stable, `<= 10` | 11 / 30 | 36.7% |
| Moderate, `10-25` | 6 / 30 | 20.0% |
| Large, `25-50` | 7 / 30 | 23.3% |
| Failure, `> 50` | 6 / 30 | 20.0% |
| Catastrophic, `> 250` | 5 / 30 | 16.7% |

Presentation interpretation:

- About one third of masked cases stayed very close to the clean prediction.
- About one fifth became clear failures.
- Most failures were catastrophic rather than mildly above threshold, which reinforces the heavy-tail behavior.

## Results By Fill Mode

| Fill mode | Count | Mean drift | Median drift | Max drift |
|---|---:|---:|---:|---:|
| `zero` | 10 | 132.44 | 45.32 | 375.60 |
| `half` | 10 | 53.68 | 15.45 | 370.60 |
| `interp` | 10 | 51.98 | 8.40 | 411.60 |

Interpretation:

- `zero` filling is the most damaging on average.
- `interp` has the lowest median drift, meaning it usually preserves the model output best.
- However, even `interp` can fail badly at very high missing area, especially near 69% censorship.
- `half` is intermediate: often stable, but one high-censorship case fails strongly.

Quantified rates by fill mode:

| Fill mode | Stable `<=10` | Failure `>50` | Catastrophic `>250` |
|---|---:|---:|---:|
| `zero` | 2 / 10, 20.0% | 4 / 10, 40.0% | 3 / 10, 30.0% |
| `half` | 3 / 10, 30.0% | 1 / 10, 10.0% | 1 / 10, 10.0% |
| `interp` | 6 / 10, 60.0% | 1 / 10, 10.0% | 1 / 10, 10.0% |

The clearest fill-mode conclusion is that interpolation was most stable in this example run, while zero-fill had the highest failure rate.

## Results By Censored Area

| Actual censored range | Count | Mean drift | Median drift | Max drift |
|---|---:|---:|---:|---:|
| 10-30% | 11 | 40.66 | 7.56 | 375.60 |
| 30-50% | 10 | 67.73 | 18.44 | 373.86 |
| 50-70% | 8 | 156.58 | 45.32 | 411.60 |

Interpretation:

- Drift generally increases as censored area increases.
- The Pearson correlation between censored ratio and drift is about `0.43`, so the relationship is positive but not purely linear.
- Mask location matters. One 22.6% zero-filled mask caused a severe failure, while some larger masks produced moderate drift.

Quantified rates by censored-area range:

| Actual censored range | Stable `<=10` | Failure `>50` | Catastrophic `>250` |
|---|---:|---:|---:|
| 10-30% | 8 / 11, 72.7% | 1 / 11, 9.1% | 1 / 11, 9.1% |
| 30-50% | 2 / 10, 20.0% | 2 / 10, 20.0% | 1 / 10, 10.0% |
| 50-70% | 0 / 8, 0.0% | 3 / 8, 37.5% | 3 / 8, 37.5% |

This gives a simple before/after story:

- Below 30% missing area, most cases stayed stable.
- From 30-50%, stability dropped sharply.
- Above 50%, none of the tested masks stayed in the stable category.

## Degradation Rate

A linear fit over all 30 runs gives an approximate increase of:

```text
33.45 drift units per additional 10% censored area
```

However, this slope is inflated by catastrophic outliers. If the five catastrophic runs above `250` drift are excluded, the fitted increase is:

```text
8.02 drift units per additional 10% censored area
```

Interpretation:

The model has two behaviors:

- Normal degradation: drift rises gradually as more pixels are hidden.
- Failure mode: certain masks cause sudden large jumps that dominate the average.

For a professor-facing discussion, this is stronger than simply saying "error increases." The result suggests that missing data creates a risk of abrupt alignment failure, not only smooth accuracy degradation.

## Threshold-Based Takeaway

Using `drift > 50` as the smoke-test failure threshold:

| Condition | Failure rate |
|---|---:|
| No censorship | 0.0% |
| All censored variants | 20.0% |
| Zero-fill masks | 40.0% |
| Half-fill masks | 10.0% |
| Interpolation masks | 10.0% |
| 10-30% missing | 9.1% |
| 30-50% missing | 20.0% |
| 50-70% missing | 37.5% |

Using the stricter catastrophic threshold `drift > 250`:

| Condition | Catastrophic rate |
|---|---:|
| No censorship | 0.0% |
| All censored variants | 16.7% |
| Zero-fill masks | 30.0% |
| Half-fill masks | 10.0% |
| Interpolation masks | 10.0% |
| 10-30% missing | 9.1% |
| 30-50% missing | 10.0% |
| 50-70% missing | 37.5% |

The practical point: the model should not blindly accept predictions from heavily censored inputs. A quality gate or uncertainty/rejection mechanism is justified.

## Notable Failure Cases

The largest prediction drifts were:

| Variation | Fill | Actual censored area | Drift |
|---:|---|---:|---:|
| 29 | `interp` | 69.1% | 411.60 |
| 6 | `zero` | 22.6% | 375.60 |
| 15 | `zero` | 40.8% | 373.86 |
| 25 | `half` | 61.8% | 370.60 |
| 24 | `zero` | 59.8% | 287.82 |

What this suggests:

- Very large missing regions can break the model regardless of fill strategy.
- Zero masks can be dangerous even at moderate missing area, likely because black rectangles are out-of-distribution artifacts.
- The model is not only sensitive to how much is missing; it is also sensitive to where the missing region lands.

## Stable Cases

The most stable cases were:

| Variation | Fill | Actual censored area | Drift |
|---:|---|---:|---:|
| 2 | `interp` | 14.2% | 0.57 |
| 5 | `interp` | 20.5% | 2.26 |
| 11 | `interp` | 25.6% | 2.60 |
| 0 | `zero` | 10.0% | 3.05 |
| 1 | `half` | 9.6% | 3.75 |

What this suggests:

- Low censorship is often tolerated.
- Interpolation tends to be the least disruptive when the missing area is small to moderate.

## Threshold View

Fraction of runs exceeding drift thresholds:

| Drift threshold | Count | Fraction |
|---:|---:|---:|
| >= 10 | 19 / 30 | 63.3% |
| >= 25 | 13 / 30 | 43.3% |
| >= 50 | 6 / 30 | 20.0% |
| >= 100 | 6 / 30 | 20.0% |
| >= 250 | 5 / 30 | 16.7% |

Interpretation:

The model often changes somewhat under masking, but catastrophic changes are concentrated in a smaller subset of cases.

## How To Explain This To The Professor

Use this wording:

"Since the official Hugging Face dataset was temporarily inaccessible, I first validated the full experimental pipeline using the repository's bundled example pair. I generated missing-data masks from roughly 10% to 70% of the thermal input and measured how much the predicted homography changed relative to the clean prediction. This is not the final ground-truth MACE result, but it confirms that the model, masking code, logging, and plotting all work in Colab. The preliminary behavior shows a heavy-tail failure pattern: many masks are tolerated, but some masks cause very large homography shifts, especially zero-filled masks and high missing-area cases."

## Technical Explanation

STHN estimates homography through dense feature matching between satellite and thermal images. Censoring removes or corrupts local evidence in the thermal image. If the remaining visible structures still provide enough geometric cues, the model output stays close to the clean prediction. If the mask covers critical structures or creates artificial features, the correlation volume can become misleading, and the iterative updater can converge to a very different four-corner displacement.

This explains why the result is not controlled only by percentage missing. A smaller mask over important visual structure can be worse than a larger mask over less informative regions.

## What The Result Supports

This run supports these claims:

- The Colab deployment is working.
- The pretrained model can run inference.
- The missing-data mask generation works.
- The script can produce CSV and plot outputs.
- Missing data can cause both gradual degradation and abrupt failures.
- Fill strategy matters: zero-fill is riskier on average than interpolation.

This run does not support these claims yet:

- It does not measure official test-set MACE.
- It does not prove generalization across the STHN dataset.
- It does not evaluate uncertainty from a trained uncertainty head.
- It does not establish a final censorship threshold for deployment.

## Next Step For Final Results

When the official dataset becomes available, run:

```bash
python experiments/missing_data_eval.py \
  --datasets_folder <DATASETS_ROOT> \
  --dataset_name satellite_0_thermalmapping_135_train \
  --split test \
  --two_stages \
  --batch_size 1 \
  --num_variations 100 \
  --mask_target thermal \
  --fill_modes zero half interp
```

That final run will compute true MACE and center error because the dataset loader provides ground-truth flow/homography targets.

## Suggested Presentation Slide Structure

1. Motivation:
   Test how STHN behaves when test-time pixels are missing or censored.

2. Method:
   Apply rectangular masks to the thermal input, fill with zero, 0.5, or interpolation, then run the pretrained model without retraining.

3. Current status:
   Full Colab pipeline is deployed and verified on the repository example pair.

4. Metric used in smoke test:
   Prediction drift relative to the clean prediction, because numeric ground truth is unavailable for the examples.

5. Main observation:
   Robustness is heavy-tailed. Most cases show moderate drift, but a subset causes severe prediction jumps.

6. Practical implication:
   A deployment system should detect missing-data conditions and possibly prefer interpolation or quality-based rejection over raw zero-filled masks.

7. Next step:
   Run the same experiment on the official test set to compute true MACE/CE once dataset access is available.
