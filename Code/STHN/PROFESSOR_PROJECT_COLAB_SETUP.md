# Colab Setup: Professor Missing-Data Workflow For STHN

## 1. What The Professor Is Asking For

The transcript points to a practical test of the existing STHN homography model under incomplete observations.

The model receives:

- a thermal/query observation, usually processed around `512x512` then resized internally;
- a larger satellite/map crop, commonly around `1536x1536` for the two-stage model;
- it predicts the four corner displacements / homography alignment.

The professor's requested first experiment is:

1. Do **not** retrain at first.
2. Take existing test images.
3. Artificially remove/censor contiguous image regions, not random pixel noise.
4. Use simple missing-data patterns, such as 1 or 2 rectangular blocks/strips.
5. Keep scale and rotation controlled.
6. Fill missing regions using simple choices:
   - zero,
   - constant middle value like `0.5`,
   - basic interpolation/inpainting.
7. Feed the censored image to the pretrained model.
8. Measure how the predicted homography changes or, if the official dataset is available, measure true MACE/CE error.

The professor also emphasized that constant filling can introduce artificial edges. Therefore, interpolation/inpainting should be tested as a simple baseline, but the likely stronger future direction is retraining with explicit missingness/mask information.

## 2. Colab Runtime

In Colab:

```text
Runtime -> Change runtime type -> GPU
```

T4 is enough for the example/smoke test. For the two-stage model, use batch size `1`.

## 3. Mount Drive And Enter The Repo

```python
from google.colab import drive
drive.mount('/content/drive')
```

Find the repo if needed:

```bash
!find /content/drive/MyDrive -name "STHN_demo.py" -print
```

Then enter your repo root. Adjust the path if your folder has a different name:

```bash
%cd /content/drive/MyDrive/STHN-main
!pwd
!ls
```

You should see:

```text
STHN_demo.py
local_pipeline/
experiments/
scripts/
examples/
```

## 4. Install Dependencies

Do not use a broad `-U` upgrade in Colab because it can upgrade pandas to a version incompatible with preinstalled Colab packages.

```bash
!pip install -q "pandas==2.2.2" kornia h5py scikit-image matplotlib opencv-python-headless \
  transformers huggingface_hub safetensors timm einops prettytable wandb faiss-cpu
```

Check:

```python
import torch, kornia, h5py, pandas
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("kornia:", kornia.__version__)
print("pandas:", pandas.__version__)
```

## 5. Hugging Face Login

The pretrained model downloads from Hugging Face. A read token is enough.

```python
from huggingface_hub import login
login()
```

Paste your Hugging Face **read** token.

## 6. Quick Model Smoke Test

Run the original demo first:

```bash
!python STHN_demo.py --two_stages
```

Expected output:

```text
Visualization saved to examples/STHN_result_two_stage.png
```

## 7. Professor-Style Missing-Data Smoke Test Without Dataset

This uses the original repo example pair and creates many censored variants of the thermal input.

This is useful when the official dataset is too large or unavailable.

Run a professor-aligned range first: missing area up to about one third of the image.

```bash
!python experiments/missing_data_examples_demo.py \
  --two_stages \
  --num_variations 100 \
  --min_ratio 0.05 \
  --max_ratio 0.33 \
  --mask_target thermal \
  --fill_modes zero half interp
```

What this means:

- `--num_variations 100`: creates 100 censored versions of the same example pair.
- `--min_ratio 0.05`: start at 5% missing area.
- `--max_ratio 0.33`: stop around one third missing area.
- `--mask_target thermal`: censor the observation/query image.
- `--fill_modes zero half interp`: compare constant and interpolation-based filling.

Important:

This does **not** produce true MACE/CE because the bundled examples do not include numeric homography ground truth. It measures prediction drift versus the clean prediction.

Display the result:

```python
import glob
import pandas as pd
from IPython.display import Image, display

latest = sorted(glob.glob("/content/drive/MyDrive/STHN-main/outputs/missing_data_examples_*"))[-1]
print(latest)

df = pd.read_csv(f"{latest}/missing_data_examples_results.csv")
display(df.head())
display(df.describe())
display(Image(f"{latest}/prediction_drift_vs_censored_ratio.png"))
display(Image(f"{latest}/clean_prediction.png"))
```

## 8. Stress Test Range

After the controlled range works, you can stress the model up to 70% missing area:

```bash
!python experiments/missing_data_examples_demo.py \
  --two_stages \
  --num_variations 100 \
  --min_ratio 0.10 \
  --max_ratio 0.70 \
  --mask_target thermal \
  --fill_modes zero half interp
```

Use this as a robustness stress test, not as the primary professor-aligned first result.

## 9. Full Dataset / H5 Version

If you have a full or minimal STHN test subset, use:

```bash
!python experiments/missing_data_eval.py \
  --datasets_folder /content/drive/MyDrive/STHN_DATASETS/minimal \
  --dataset_name satellite_0_thermalmapping_135_train \
  --split test \
  --two_stages \
  --batch_size 1 \
  --num_variations 100 \
  --min_ratio 0.05 \
  --max_ratio 0.33 \
  --mask_target thermal \
  --fill_modes zero half interp
```

This version computes true MACE/CE because it uses dataset ground truth.

Expected minimal dataset layout:

```text
/content/drive/MyDrive/STHN_DATASETS/minimal/
  maps/satellite/20201117_BingSatellite.png
  satellite_0_thermalmapping_135_train/test_queries.h5
  satellite_0_thermalmapping_135_train/test_database.h5
```

## 10. Recommended Result To Report

For a first professor update:

1. Run the controlled missing range `0.05-0.33`.
2. Report drift vs clean prediction if using examples.
3. Report true MACE/CE if using `.h5` dataset.
4. Compare fill methods: `zero`, `half`, `interp`.
5. Show masked examples and the drift/error plot.
6. Clearly state whether the run used:
   - one example pair with 100 variants, or
   - 100 real dataset samples.

## 11. Common Colab Errors

### `ModuleNotFoundError: No module named 'kornia'`

Run the dependency install cell again.

### Hugging Face rate-limit warning

Login with a read token:

```python
from huggingface_hub import login
login()
```

### Out of memory

Use:

```text
--batch_size 1
```

For the example demo, batch size is already effectively one.

### Dataset too large

Do not download the full 122-131 GB dataset into a 12 GB Drive. Ask for a minimal test subset or use the example-based smoke test.

