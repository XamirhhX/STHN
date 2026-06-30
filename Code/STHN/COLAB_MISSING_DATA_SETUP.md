# Google Colab Setup For The STHN Missing-Data Experiment

Use a GPU runtime first: `Runtime -> Change runtime type -> T4/A100 GPU`.

Important: the dataset is not inside the repository. The repository contains
code, pretrained-demo loading logic, and small `examples/` images only. The full
STHN/Boson-nighttime dataset must be downloaded separately from Hugging Face and
requires you to accept the dataset access terms while logged in.

## 1. Upload This Edited Repository

The missing-data scripts in this workspace are not part of the upstream repository
unless you commit or copy them there. The safest Colab path is to upload this
edited project folder to Google Drive, then mount it:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/STHN-main
```

If you clone the upstream repository instead, also copy these files into the clone:

```text
experiments/missing_data_eval.py
experiments/missing_data_examples_demo.py
scripts/check_sthn_hardware.py
COLAB_MISSING_DATA_SETUP.md
```

## 2. Install Dependencies

Colab already provides Python and CUDA-capable PyTorch in many runtimes. This cell installs the repo-specific packages.
Do not use a broad `-U` upgrade here because Colab pins `pandas==2.2.2` and upgrading to pandas 3.x breaks several preinstalled Colab packages.

```bash
!pip install -q "pandas==2.2.2" kornia h5py scikit-image matplotlib opencv-python-headless \
  transformers huggingface_hub safetensors timm einops prettytable wandb faiss-cpu
```

If you already ran a command that installed pandas 3.x, repair the runtime with:

```bash
!pip install -q "pandas==2.2.2"
```

Then restart the runtime once: `Runtime -> Restart runtime`.

If PyTorch is missing or CPU-only, install CUDA wheels explicitly:

```bash
!pip install -q --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

## 3. Mount The Dataset

The full STHN dataset is large, so keep it on Google Drive or another mounted storage path:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create a target folder:

```bash
!mkdir -p /content/drive/MyDrive/STHN_DATASETS
```

Then download the dataset from:

```text
https://huggingface.co/datasets/xjh19972/boson-nighttime/tree/main/satellite-thermal-dataset-v3
```

Because this is a gated dataset, first open that URL in your browser, log in to
Hugging Face, and accept the access conditions. Then create a Hugging Face token:

```text
https://huggingface.co/settings/tokens
```

Recommended token for this project:

- Token kind: User Access Token
- Role/type: `read`
- Name: `colab-sthn-read`
- Do not create a `write` token for this workflow.

More secure alternative:

- Role/type: `fine-grained`
- Grant read access to:
  - dataset `xjh19972/boson-nighttime`
  - model `xjh19972/STHN`

If the fine-grained token gives a 403 error, use a normal `read` token. The token only works after your Hugging Face account has been granted access to the gated dataset in the browser.

In Colab:

```bash
!pip install -q huggingface_hub hf_transfer
```

```python
from huggingface_hub import login
login()  # paste your Hugging Face token
```

Download the multipart archive into Drive:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="xjh19972/boson-nighttime",
    repo_type="dataset",
    allow_patterns=[
        "satellite-thermal-dataset-v3/satellite_thermal_dataset_v3.tar.gz.part*",
    ],
    local_dir="/content/drive/MyDrive/STHN_DATASETS/hf_download",
    local_dir_use_symlinks=False,
)
```

Concatenate and extract:

```bash
%cd /content/drive/MyDrive/STHN_DATASETS/hf_download/satellite-thermal-dataset-v3
!cat satellite_thermal_dataset_v3.tar.gz.part* > satellite_thermal_dataset_v3.tar.gz
!mkdir -p /content/drive/MyDrive/STHN_DATASETS/extracted
!tar -xzf satellite_thermal_dataset_v3.tar.gz -C /content/drive/MyDrive/STHN_DATASETS/extracted
```

After extraction, locate the folder that contains `maps/` and the
`satellite_0_thermalmapping_135*` folders:

```bash
!find /content/drive/MyDrive/STHN_DATASETS/extracted -maxdepth 4 -type d -name maps -print
!find /content/drive/MyDrive/STHN_DATASETS/extracted -maxdepth 4 -type f -name test_queries.h5 | head
```

Expected final layout under whichever extracted folder is your datasets root:

```text
/content/drive/MyDrive/STHN_DATASETS/extracted/<maybe-one-folder>/datasets/
  maps/satellite/20201117_BingSatellite.png
  satellite_0_thermalmapping_135_train/test_database.h5
  satellite_0_thermalmapping_135_train/test_queries.h5
```

## 4. Check GPU Memory

```bash
!python scripts/check_sthn_hardware.py --two_stages --batch_size 1
```

Use `--batch_size 1` for the two-stage model on smaller GPUs. Increase only after the check shows memory headroom.

## 5. Dataset-Free Smoke Test

If the Hugging Face dataset is unavailable, test the deployed code with the bundled
example pair:

```bash
!python experiments/missing_data_examples_demo.py \
  --two_stages \
  --num_variations 30 \
  --mask_target thermal \
  --fill_modes zero half interp
```

This still downloads the pretrained model. If model download is also unavailable,
run a pure execution check with random weights:

```bash
!python experiments/missing_data_examples_demo.py \
  --random_weights \
  --num_variations 5
```

The random-weights run is only a code/runtime test; do not use its numbers in a report.

## 6. Run The Missing-Data Experiment

```bash
!python experiments/missing_data_eval.py \
  --datasets_folder /content/drive/MyDrive/STHN_DATASETS/extracted/<path-to-datasets-root> \
  --dataset_name satellite_0_thermalmapping_135_train \
  --split test \
  --two_stages \
  --batch_size 1 \
  --num_variations 100 \
  --mask_target thermal \
  --fill_modes zero half interp \
  --uncertainty_repeats 1
```

Outputs are written under `outputs/missing_data_<timestamp>/`:

- `missing_data_results.csv`
- `error_vs_censored_ratio.png`
- `summary.json`
- a few masked input examples in `examples/`
