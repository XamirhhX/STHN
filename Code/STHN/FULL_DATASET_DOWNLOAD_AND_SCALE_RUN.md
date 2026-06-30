# Download Full STHN Dataset To Google Drive And Run Scale Task

## 1. What To Download

The STHN/Boson-nighttime dataset is separate from the code repository.

Dataset page:

```text
https://huggingface.co/datasets/xjh19972/boson-nighttime/tree/main/satellite-thermal-dataset-v3
```

The relevant archive for this project is:

```text
satellite_thermal_dataset_v3.tar.gz.part*
```

Do not download `extended_queries_test_excluded.tar.gz.part*` for the first scale experiment. That archive is for the generated/extended training data protocol, not required for test-time scale evaluation.

## 2. Prepare Colab

Use a GPU runtime:

```text
Runtime -> Change runtime type -> GPU
```

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Install dependencies:

```bash
!pip install -q "pandas==2.2.2" kornia h5py scikit-image matplotlib opencv-python-headless \
  transformers safetensors timm einops prettytable wandb faiss-cpu \
  "huggingface_hub[hf_xet]" hf_transfer
```

Login:

```python
from huggingface_hub import login
login()
```

Paste a Hugging Face **read** token.

## 3. Download The Dataset Archive Parts

Create folders:

```bash
!mkdir -p /content/drive/MyDrive/STHN_DATASETS/hf_download
!mkdir -p /content/drive/MyDrive/STHN_DATASETS/extracted
```

Download only the main dataset archive parts:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="xjh19972/boson-nighttime",
    repo_type="dataset",
    allow_patterns=[
        "satellite-thermal-dataset-v3/satellite_thermal_dataset_v3.tar.gz.part*",
        "satellite-thermal-dataset-v3/README.md",
    ],
    local_dir="/content/drive/MyDrive/STHN_DATASETS/hf_download",
)
```

## 4. Combine The Multipart Archive

```bash
%cd /content/drive/MyDrive/STHN_DATASETS/hf_download/satellite-thermal-dataset-v3
!ls -lh satellite_thermal_dataset_v3.tar.gz.part*
!cat satellite_thermal_dataset_v3.tar.gz.part* > satellite_thermal_dataset_v3.tar.gz
!ls -lh satellite_thermal_dataset_v3.tar.gz
```

Expected archive size is roughly 130 GB.

## 5. Extract The Dataset

```bash
!tar -xzf satellite_thermal_dataset_v3.tar.gz -C /content/drive/MyDrive/STHN_DATASETS/extracted
```

This can take a long time on Google Drive.

## 6. Find The Dataset Root

```bash
!find /content/drive/MyDrive/STHN_DATASETS/extracted -maxdepth 6 -type f -name test_queries.h5 | head -20
!find /content/drive/MyDrive/STHN_DATASETS/extracted -maxdepth 6 -type f -name 20201117_BingSatellite.png
```

The dataset root is the folder containing both:

```text
maps/satellite/20201117_BingSatellite.png
satellite_0_thermalmapping_135_train/test_queries.h5
```

Example root:

```text
/content/drive/MyDrive/STHN_DATASETS/extracted/datasets
```

Use your actual path as `<DATASETS_ROOT>`.

## 7. Inspect The H5 Files

Go to repo root:

```bash
%cd /content/drive/MyDrive/STHN-main
```

Inspect queries:

```bash
!python scripts/inspect_sthn_h5.py \
  --h5 <DATASETS_ROOT>/satellite_0_thermalmapping_135_train/test_queries.h5
```

Inspect database:

```bash
!python scripts/inspect_sthn_h5.py \
  --h5 <DATASETS_ROOT>/satellite_0_thermalmapping_135_train/test_database.h5
```

## 8. Run The Scale Task On H5 Dataset

Professor's fixed 20% closer case:

```bash
!python experiments/scaled_observation_eval.py \
  --datasets_folder <DATASETS_ROOT> \
  --dataset_name satellite_0_thermalmapping_135_train \
  --split test \
  --two_stages \
  --batch_size 1 \
  --num_workers 0 \
  --num_variations 100 \
  --closer_percent 20 \
  --num_tiles 2 \
  --path diagonal \
  --fill_modes zero half mean
```

Outputs:

```text
outputs/scaled_observation_h5_<timestamp>/
  scaled_observation_h5_results.csv
  mace_vs_missing_ratio.png
  mace_vs_scale.png
  examples/
```

## 9. Run A Scale Sweep

This tests normal scale to 40% smaller thermal footprint:

```bash
!python experiments/scaled_observation_eval.py \
  --datasets_folder <DATASETS_ROOT> \
  --dataset_name satellite_0_thermalmapping_135_train \
  --split test \
  --two_stages \
  --batch_size 1 \
  --num_workers 0 \
  --num_variations 100 \
  --min_scale 1.0 \
  --max_scale 0.6 \
  --num_tiles 2 \
  --path diagonal \
  --fill_modes zero half mean
```

## 10. Show Results In Colab

```python
import glob
import pandas as pd
from IPython.display import Image, display

latest = sorted(glob.glob("/content/drive/MyDrive/STHN-main/outputs/scaled_observation_h5_*"))[-1]
print(latest)

df = pd.read_csv(f"{latest}/scaled_observation_h5_results.csv")
display(df.head())
display(df.describe())

display(Image(f"{latest}/mace_vs_missing_ratio.png"))
display(Image(f"{latest}/mace_vs_scale.png"))
```

## 11. Optional: Create A Compact 100-Query Subset

After extraction, if you want a smaller portable subset:

```bash
%cd /content/drive/MyDrive/STHN-main

!python scripts/create_minimal_sthn_subset.py \
  --source_datasets <DATASETS_ROOT> \
  --output_datasets /content/drive/MyDrive/STHN_DATASETS/STHN_minimal_100 \
  --dataset_name satellite_0_thermalmapping_135_train \
  --split test \
  --num_queries 100
```

Then use:

```text
--datasets_folder /content/drive/MyDrive/STHN_DATASETS/STHN_minimal_100
```

