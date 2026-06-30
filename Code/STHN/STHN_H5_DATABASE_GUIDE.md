# Guide: How To Use The STHN `.h5` Dataset Files

## 1. What The `.h5` Files Are

The STHN dataset stores query/database metadata and image arrays in HDF5 files.

Typical layout:

```text
datasets/
  maps/
    satellite/
      20201117_BingSatellite.png
  satellite_0_thermalmapping_135_train/
    test_queries.h5
    test_database.h5
    train_queries.h5
    train_database.h5
    val_queries.h5
    val_database.h5
```

For the local homography model:

- `test_queries.h5` contains thermal/query images and their names/coordinates.
- `test_database.h5` contains database/map crop names and their coordinates.
- `maps/satellite/20201117_BingSatellite.png` is the large satellite map from which database crops are extracted.

In this repo's `local_pipeline/datasets_4cor_img.py`, the database H5 is mainly used for `image_name` and UTM coordinates. The satellite image itself is cropped from the large map PNG.

## 2. Minimal Test Subset

For professor's missing-data test, you do not need the full dataset.

A minimal test subset can be:

```text
STHN_minimal_100/
  maps/satellite/20201117_BingSatellite.png
  satellite_0_thermalmapping_135_train/test_queries.h5
  satellite_0_thermalmapping_135_train/test_database.h5
```

Then pass:

```text
--datasets_folder /path/to/STHN_minimal_100
--dataset_name satellite_0_thermalmapping_135_train
--split test
```

## 3. Inspect A Query H5 File

Run in Colab or local Python:

```python
import h5py

path = "/content/drive/MyDrive/STHN_DATASETS/minimal/satellite_0_thermalmapping_135_train/test_queries.h5"

with h5py.File(path, "r") as f:
    print("keys:", list(f.keys()))
    for key in f.keys():
        obj = f[key]
        print(key, obj.shape, obj.dtype)
    print("num query rows:", len(f["image_name"]))
    print("first names:")
    for name in f["image_name"][:5]:
        print(name.decode("utf-8"))
```

Common keys:

```text
image_name
image_data
```

`image_name` encodes position metadata. The repo parses UTM-like coordinates from the name by splitting on `@`.

## 4. Inspect A Database H5 File

```python
import h5py

path = "/content/drive/MyDrive/STHN_DATASETS/minimal/satellite_0_thermalmapping_135_train/test_database.h5"

with h5py.File(path, "r") as f:
    print("keys:", list(f.keys()))
    for key in f.keys():
        obj = f[key]
        print(key, obj.shape, obj.dtype)
    print("num database rows:", len(f["image_name"]))
    for name in f["image_name"][:5]:
        print(name.decode("utf-8"))
```

For this repo's local pipeline, `test_database.h5` may only need `image_name`, because the actual satellite crop is extracted from `maps/satellite/20201117_BingSatellite.png`.

## 5. Decode Coordinates From `image_name`

The loader assumes names contain coordinate fields separated by `@`.

Example:

```python
def parse_utm(name):
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    parts = name.split("@")
    easting = float(parts[1])
    northing = float(parts[2])
    return easting, northing

with h5py.File(path, "r") as f:
    name = f["image_name"][0]
    print(name)
    print(parse_utm(name))
```

The dataset class uses these coordinates to compute the ground-truth translation/flow between query and database images.

## 6. Visualize A Query Image From H5

```python
import h5py
from PIL import Image
import matplotlib.pyplot as plt

query_h5 = "/content/drive/MyDrive/STHN_DATASETS/minimal/satellite_0_thermalmapping_135_train/test_queries.h5"

with h5py.File(query_h5, "r") as f:
    img = f["image_data"][0]
    name = f["image_name"][0].decode("utf-8")

print(name)
print(img.shape, img.dtype)

plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
```

If the image is RGB, remove `cmap="gray"`.

## 7. Visualize A Satellite Crop From The Map

The repo does this internally, but this cell shows the idea.

```python
from PIL import Image
import matplotlib.pyplot as plt
import h5py

datasets_folder = "/content/drive/MyDrive/STHN_DATASETS/minimal"
dataset_name = "satellite_0_thermalmapping_135_train"
database_size = 1536

db_h5 = f"{datasets_folder}/{dataset_name}/test_database.h5"
map_path = f"{datasets_folder}/maps/satellite/20201117_BingSatellite.png"

with h5py.File(db_h5, "r") as f:
    name = f["image_name"][0].decode("utf-8")

parts = name.split("@")
easting = float(parts[1])
northing = float(parts[2])

sat_map = Image.open(map_path).convert("RGB")

left = int(northing) - database_size // 2
top = int(easting) - database_size // 2
right = int(northing) + database_size // 2
bottom = int(easting) + database_size // 2

crop = sat_map.crop((left, top, right, bottom))

plt.figure(figsize=(5, 5))
plt.imshow(crop)
plt.axis("off")
plt.show()
```

Note: the repo swaps coordinate order in `datasets_4cor_img.py`, so if a manual crop looks wrong, inspect that file's `_find_img_in_map` implementation and follow it exactly.

## 8. Use The Repo DataLoader Directly

From the STHN repo root:

```python
import sys
from types import SimpleNamespace

sys.path.insert(0, "local_pipeline")
import datasets_4cor_img as datasets

args = SimpleNamespace(
    datasets_folder="/content/drive/MyDrive/STHN_DATASETS/minimal",
    dataset_name="satellite_0_thermalmapping_135_train",
    batch_size=1,
    num_workers=0,
    augment="none",
    eval_model="pretrained_hf",
    perspective_max=0,
    rotate_max=0,
    resize_max=0,
    crop_width=512,
    resize_width=256,
    database_size=1536,
    val_positive_dist_threshold=50,
    prior_location_threshold=-1,
    G_contrast="none",
    load_test_pairs=None,
    generate_test_pairs=False,
)

loader = datasets.fetch_dataloader(args, split="test")

batch = next(iter(loader))
satellite, thermal, flow_gt, H, query_utm, database_utm, index, pos_index = batch

print("satellite:", satellite.shape)
print("thermal:", thermal.shape)
print("flow_gt:", flow_gt.shape)
print("H:", H.shape)
print("query_utm:", query_utm)
print("database_utm:", database_utm)
```

Expected approximate shapes for two-stage setup:

```text
satellite: [1, 3, 1536, 1536]
thermal:   [1, 3, 256, 256]
flow_gt:   [1, 2, 256, 256]
H:         [1, 3, 3]
```

## 9. Run The Missing-Data Evaluation On H5

```bash
python experiments/missing_data_eval.py \
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

Outputs:

```text
outputs/missing_data_<timestamp>/
  missing_data_results.csv
  error_vs_censored_ratio.png
  summary.json
  examples/
```

## 10. How Ground Truth Is Computed

The H5 files do not necessarily store final MACE directly.

The loader computes the target homography/flow from:

1. query UTM coordinate,
2. database UTM coordinate,
3. resize/database scale,
4. expected crop geometry.

Then `missing_data_eval.py` compares:

```text
model predicted four corners
vs
ground-truth four corners from flow_gt
```

This gives MACE and center error.

## 11. Common H5 Problems

### `FileNotFoundError`

Check that your folder has:

```text
maps/satellite/20201117_BingSatellite.png
satellite_0_thermalmapping_135_train/test_queries.h5
satellite_0_thermalmapping_135_train/test_database.h5
```

### No positive database match

The loader finds database positives within:

```text
--val_positive_dist_threshold 50
```

If your mini subset has only query rows but not enough database rows, the loader may fail. Keep all database `image_name` rows or use a subset creator that preserves positive matches.

### Colab memory issue

Use:

```text
--batch_size 1
--num_workers 0
```

### H5 image display looks wrong

Inspect:

```python
img.shape
img.dtype
img.min()
img.max()
```

Some arrays are grayscale, some are RGB, and some may be uint8.

## 12. Creating A 100-Query Minimal H5 Subset

If a machine has the full dataset, run:

```bash
python scripts/create_minimal_sthn_subset.py \
  --source_datasets /path/to/full/STHN/datasets \
  --output_datasets /path/to/STHN_minimal_100 \
  --dataset_name satellite_0_thermalmapping_135_train \
  --split test \
  --num_queries 100
```

Then zip and send:

```text
STHN_minimal_100/
  maps/satellite/20201117_BingSatellite.png
  satellite_0_thermalmapping_135_train/test_queries.h5
  satellite_0_thermalmapping_135_train/test_database.h5
```

