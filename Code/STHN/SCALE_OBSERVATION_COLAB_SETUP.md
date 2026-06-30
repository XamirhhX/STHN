# Colab Setup: Scaled Partial Thermal Observations

## 1. Goal

This setup is for the updated project direction:

> The issue is no longer simple censorship. We want to simulate drone altitude/scale change. For example, if the drone is 20% closer to Earth, the thermal observation footprint is treated as 20% smaller. Multiple such smaller thermal observations are placed into one standard input canvas, leaving no-data regions where no observation exists.

This creates a stitched/mosaic-like thermal input:

```text
standard thermal canvas
  contains several smaller thermal observations
  plus blank/no-data regions
```

The model is then tested without retraining first.

## 2. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Go to your STHN repo:

```bash
%cd /content/drive/MyDrive/STHN-main
!pwd
!ls
```

## 3. Install Dependencies

```bash
!pip install -q "pandas==2.2.2" kornia h5py scikit-image matplotlib opencv-python-headless \
  transformers huggingface_hub safetensors timm einops prettytable wandb faiss-cpu
```

Check:

```python
import torch, kornia
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print(kornia.__version__)
```

## 4. Hugging Face Login

```python
from huggingface_hub import login
login()
```

Use a Hugging Face **read** token.

## 5. Confirm The New Scale Scripts Exist

```bash
!ls experiments/scaled_observation_examples_demo.py
!ls experiments/scaled_observation_eval.py
!ls experiments/scaled_observation_utils.py
```

## 6. Quick Demo: 20% Closer, Two Observations

This treats "20% closer" as a linear footprint scale of `0.8`.

```bash
!python experiments/scaled_observation_examples_demo.py \
  --two_stages \
  --num_variations 30 \
  --closer_percent 20 \
  --num_tiles 2 \
  --path diagonal \
  --fill_modes zero half mean
```

Outputs are written to:

```text
outputs/scaled_observation_examples_<timestamp>/
```

Important output files:

```text
scaled_observation_results.csv
drift_vs_missing_ratio.png
drift_vs_scale.png
clean_prediction.png
examples/
```

Display:

```python
import glob
import pandas as pd
from IPython.display import Image, display

latest = sorted(glob.glob("/content/drive/MyDrive/STHN-main/outputs/scaled_observation_examples_*"))[-1]
print(latest)

df = pd.read_csv(f"{latest}/scaled_observation_results.csv")
display(df.head())
display(df.describe())

display(Image(f"{latest}/drift_vs_missing_ratio.png"))
display(Image(f"{latest}/drift_vs_scale.png"))
display(Image(f"{latest}/clean_prediction.png"))
```

## 7. Controlled Scale Sweep

This tests different altitude/scale conditions from normal size to 40% smaller footprint:

```bash
!python experiments/scaled_observation_examples_demo.py \
  --two_stages \
  --num_variations 100 \
  --min_scale 1.0 \
  --max_scale 0.6 \
  --num_tiles 2 \
  --path diagonal \
  --fill_modes zero half mean
```

Interpretation:

```text
scale = 1.0  -> no footprint shrink
scale = 0.8  -> 20% smaller footprint
scale = 0.6  -> 40% smaller footprint
```

The script plots prediction drift versus:

- no-data area ratio,
- observation scale.

## 8. Change The Motion Pattern

Diagonal simulated motion:

```bash
--path diagonal
```

Horizontal strips:

```bash
--path horizontal
```

Vertical strips:

```bash
--path vertical
```

Corner-style observations:

```bash
--path corners --num_tiles 4
```

Random placements:

```bash
--path random
```

## 9. More Tiles

Two partial observations:

```bash
--num_tiles 2
```

Four partial observations:

```bash
--num_tiles 4
```

More tiles usually increases coverage and reduces no-data regions, depending on overlap.

## 10. Full H5 Dataset Evaluation

If you have a minimal `.h5` test subset, run:

```bash
!python experiments/scaled_observation_eval.py \
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

This produces true dataset metrics:

```text
scaled_observation_h5_results.csv
mace_vs_missing_ratio.png
mace_vs_scale.png
```

Use the H5 version when you need real MACE/center-error results.

## 11. What To Tell The Professor

Use this wording:

> I updated the experiment from simple censorship to scaled partial observations. The new script simulates a drone altitude/scale change by shrinking the thermal observation footprint and placing multiple smaller observations into one standard input canvas. The uncovered regions are treated as no-data. The first test uses the pretrained STHN model without retraining and measures how much the predicted homography changes. When the `.h5` dataset is available, the same pipeline computes true MACE and center error.

## 12. Important Convention

In this implementation:

```text
20% closer -> scale 0.8
```

That follows the convention you described: the thermal footprint becomes 20% smaller inside the standard canvas.

