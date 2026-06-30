# Repository Structure and Code Guide

This document describes the structure of the STHN repository and explains the role of each coding/configuration file. The project implements **STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery**. The main task is to align a UAV thermal image with satellite imagery, estimate a homography, and support both coarse global retrieval and local refinement.

## High-Level Purpose

The repository contains three related pipelines:

- **Global pipeline**: image retrieval and thermal generation components used for coarse satellite-thermal matching and baseline evaluation.
- **Local pipeline**: the main STHN homography estimation pipeline, including feature extraction, correlation, iterative update blocks, training, and evaluation.
- **Keypoint pipeline**: keypoint/image-matching baselines and variants based on LoFTR-style and R2D2-style components.

The root `STHN_demo.py` script is the simplest entry point. It loads pretrained STHN weights from Hugging Face, preprocesses sample satellite/thermal inputs, predicts four-corner displacement, and visualizes the aligned result.

## Top-Level Structure

```text
STHN-main/
|-- README.md
|-- REPOSITORY_STRUCTURE.md
|-- LICENSE
|-- env.yml
|-- __init__.py
|-- STHN_demo.py
|-- train_global.sh
|-- train_local.sh
|-- eval_global.sh
|-- eval_local.sh
|-- transform_dataset.sh
|-- visualization_h5.ipynb
|-- visualization_map.ipynb
|-- cache/
|   `-- vocabulary/dinov2_vitg14/l31_value_c32/thermal/c_centers.pt
|-- examples/
|   |-- gt.png
|   |-- img1.png
|   `-- img2.png
|-- global_pipeline/
|-- local_pipeline/
|-- keypoint_pipeline/
|-- scripts/
`-- utils/
```

## Runtime and Data Assumptions

- `env.yml` defines the expected Conda environment named `STHN`.
- The project expects a `datasets/` folder at the repository root, although it is not included in this checkout.
- Training and evaluation outputs are expected under folders such as `logs/` and `test/`, which are also not included here.
- Many scripts are designed for a Slurm cluster through `.sbatch` files.
- Several modules import sibling files using non-package imports such as `import parser`, `import utils`, `import extractor`, and `import update`. Run scripts from the expected working directory or with the correct `PYTHONPATH`.

## Main Execution Flow

```text
Satellite image + thermal image
        |
        v
Preprocessing and resizing
        |
        v
Local feature extraction
        |
        v
Correlation volume construction
        |
        v
Iterative update block predicts corner displacement
        |
        v
DLT / perspective transform
        |
        v
Homography-aligned visualization or evaluation metric
```

For two-stage STHN, the coarse model first estimates a broad alignment. A crop/refinement step then estimates a finer displacement in a smaller region, and the two displacements are combined.

## Directory Guide

### `cache/`

Stores cached model artifacts. In this checkout it contains a DINOv2/AnyLoc VLAD vocabulary center tensor:

- `cache/vocabulary/dinov2_vitg14/l31_value_c32/thermal/c_centers.pt`: cached cluster centers for AnyLoc-style VLAD descriptors on thermal data.

### `examples/`

Small demo assets used by `STHN_demo.py`:

- `img1.png`: example satellite image.
- `img2.png`: example thermal image.
- `gt.png`: ground-truth or reference visualization.

### `global_pipeline/`

Contains the coarse/global image retrieval and image-to-image translation pipeline. It includes dataset loaders, retrieval models, aggregation layers, Pix2Pix translation modules, AnyLoc utilities, H5 dataset generation, training, and evaluation.

### `local_pipeline/`

Contains the main STHN local homography estimation implementation. It includes dataset handling for four-corner homography supervision, feature extractors, correlation modules, iterative update networks, training/evaluation scripts, and the trainable STHN model wrapper.

### `keypoint_pipeline/`

Contains alternative image matching baselines and keypoint-oriented pipelines. It has two main subfolders:

- `myloftr/`: LoFTR-style homography/keypoint matching components.
- `myr2d2/`: R2D2-style keypoint detection/description/training/evaluation components.

### `scripts/`

Contains Slurm `.sbatch` launchers for global retrieval, local STHN training/evaluation, larger-resolution experiments, two-stage refinement, and augmentation experiments.

### `utils/`

Small repository-level helper scripts, including H5 comparison and plotting utilities.

## Root Files

### `README.md`

Primary project README. It describes the STHN paper, citation, environment setup, demo usage, dataset layout, training procedure, evaluation procedure, and acknowledgements.

### `LICENSE`

Repository license file.

### `env.yml`

Conda environment definition. It pins Python 3.10, PyTorch 2.4, CUDA 12.1, torchvision 0.19, xformers, matplotlib, h5py, kornia, scikit-image, and pip dependencies such as `faiss-gpu`, `wandb`, `opencv-python-headless`, `transformers`, `timm`, and `einops`.

### `__init__.py`

Empty package marker for the repository root.

### `STHN_demo.py`

Standalone inference demo. It defines demo-local versions of `GMA`, `IHN`, and `STHN` so the script can load compatible checkpoints without requiring the full training argument stack. It includes preprocessing helpers for satellite RGB and thermal images, a visualization helper, Hugging Face model download/loading support, one-stage and two-stage inference, and a CLI entry point.

### `train_global.sh`

Convenience shell file listing commented Slurm commands for global retrieval training. It references NetVLAD and GeM variants, with and without DANN/domain-adversarial training.

### `train_local.sh`

Convenience shell file listing commented Slurm commands for local STHN training. It groups runs into normal, large, larger, augmented, and two-stage refinement configurations.

### `eval_global.sh`

Shell evaluation template for global retrieval models. It activates the `STHN` Conda environment and lists example `global_pipeline/eval.py` commands for NetVLAD and GeM checkpoints.

### `eval_local.sh`

Shell evaluation template for local STHN checkpoints. It activates the environment and lists example Slurm evaluation commands with different displacement-channel settings (`DC`) and optional refinement padding (`PAD`).

### `transform_dataset.sh`

Dataset generation helper. It contains example commands for `global_pipeline/h5_transformer.py` and `global_pipeline/h5_merger.py` to create satellite/thermal H5 datasets from maps, merge flights, resize crops, and optionally remove intermediate generated folders.

### `visualization_h5.ipynb`

Notebook for inspecting and visualizing H5 datasets.

### `visualization_map.ipynb`

Notebook for map-level visualization, likely used to inspect satellite/thermal map regions and splits.

## Global Pipeline Files

### `global_pipeline/README.md`

README inherited from or related to the earlier satellite-thermal geo-localization project. It explains dataset format, Thermal Generation Module training, Satellite-thermal Geo-localization Module training, and evaluation.

### `global_pipeline/LICENSE` and `global_pipeline/LICENSE_FOR_REFERENCE`

License/reference files for code used in this pipeline.

### `global_pipeline/folder_config.yml`

YAML configuration describing dataset folder names, source map names, or dataset-generation settings consumed by H5 transformation/merging utilities.

### `global_pipeline/__init__.py`

Empty package marker.

### `global_pipeline/commons.py`

Shared runtime helpers. Provides deterministic seeding through `make_deterministic` and logging setup through `setup_logging`.

### `global_pipeline/datasets_ws.py`

Large dataset module for retrieval and image-translation training. Defines `BaseDataset` for database/query inference, `PCADataset` for PCA feature extraction, `TripletsDataset` for triplet mining, `RAMEfficient2DMatrix` for sparse cached feature storage, and `TranslationDataset` for satellite-to-thermal pair generation.

### `global_pipeline/eval.py`

Evaluation launcher for global retrieval models. It parses arguments, loads retrieval checkpoints or off-the-shelf backbones, builds datasets, and calls testing utilities to compute recall metrics.

### `global_pipeline/eval_anyloc.py`

Evaluation launcher for AnyLoc/DINOv2-style global retrieval. It is parallel in role to `eval.py` but uses `test_anyloc.py` and `global_pipeline/anyloc`.

### `global_pipeline/eval_pix2pix.py`

Pix2Pix evaluation launcher for translated thermal imagery. This file currently has a Python parse issue caused by unexpected indentation around line 33, so it should be reviewed before direct execution.

### `global_pipeline/eval_pix2pix_generate_h5.py`

Launcher for generating H5 outputs with a Pix2Pix translation model. It loads model/data settings and delegates generation to `test_translation_pix2pix_generate_h5`.

### `global_pipeline/eval_pix2pix_generate_h5_exclude.py`

Variant of Pix2Pix H5 generation intended for test-excluded generated data, matching the rigorous evaluation protocol described in the root README.

### `global_pipeline/h5_merger.py`

Command-line utility for merging multiple generated or source H5 datasets. It defines `merge_h5_file`, copies/combines image arrays and metadata, supports compression/resizing options, and has a CLI entry point.

### `global_pipeline/h5_transformer.py`

Command-line utility for creating H5 datasets from large satellite/thermal maps. It computes region overlap, crops maps, stores image crops and metadata, supports stride sampling and compression, and has a CLI entry point.

### `global_pipeline/parser.py`

Argument parser for global training, evaluation, retrieval, and translation scripts. It centralizes dataset paths, model options, aggregation choices, optimization settings, PCA/cache settings, DANN settings, and Pix2Pix-related options.

### `global_pipeline/plotting.py`

Plotting helper with `process_results_simulation`, used to summarize or visualize retrieval/localization metrics.

### `global_pipeline/test.py`

Evaluation backend for global retrieval and Pix2Pix generation. It computes descriptors, performs nearest-neighbor search with FAISS, computes recall metrics, supports memory-efficient testing, evaluates translated images, generates H5 outputs, and includes top-N voting logic.

### `global_pipeline/test_anyloc.py`

Evaluation backend for AnyLoc. It extracts DINOv2 descriptors, fits or loads VLAD vocabularies, creates VLAD vectors, and evaluates retrieval performance.

### `global_pipeline/train.py`

Training loop for global retrieval models. It builds datasets and dataloaders, computes triplets, trains the retrieval network, logs metrics, saves checkpoints, and periodically evaluates.

### `global_pipeline/train_pix2pix.py`

Training loop for the Thermal Generation Module/Pix2Pix path. It trains image-to-image translation components, logs through W&B, saves checkpoints, and evaluates generated thermal outputs.

### `global_pipeline/util.py`

Global utility functions for model FLOP estimation, checkpoint saving, model resume, Pix2Pix resume, full training resume, PCA computation, and learning-rate adjustment.

## Global AnyLoc Files

### `global_pipeline/anyloc/__init__.py`

Empty package marker.

### `global_pipeline/anyloc/LICENSE`

License for the AnyLoc-derived code.

### `global_pipeline/anyloc/utilities.py`

AnyLoc/DINOv2 support code. Defines `DinoV2ExtractFeatures` for extracting intermediate DINOv2 patch descriptors and `VLAD` for fitting/loading cluster centers and generating VLAD global descriptors with hard or soft assignment.

## Global Model Files

### `global_pipeline/model/__init__.py`

Empty package marker.

### `global_pipeline/model/aggregation.py`

Image retrieval aggregation layers. Defines MAC, SPoC, GeM, RMAC, Flatten, RRM, NetVLAD, and context reweighting modules (`CRNModule`, `CRN`) for converting convolutional feature maps into global descriptors.

### `global_pipeline/model/Deit.py`

DeiT distilled Vision Transformer wrappers. Defines `DistilledVisionTransformer` and constructors for small/base distilled DeiT variants.

### `global_pipeline/model/functional.py`

Functional primitives for retrieval and training losses. Includes domain-adversarial gradient reversal (`ReverseLayerF`), SARE losses, pooling functions (`mac`, `spoc`, `gem`, `rmac`), and PSNR computation.

### `global_pipeline/model/network.py`

Main global model factory and network definitions. Defines `GeoLocalizationNet`, `pix2pix`, and `GeoLocalizationNetRerank`, plus helper functions for choosing aggregations, pretrained models, backbones, and output channel dimensions.

### `global_pipeline/model/non_local.py`

Defines `NonLocalBlock`, a non-local attention/context block used to enhance feature interactions.

### `global_pipeline/model/normalization.py`

Defines `L2Norm`, a small module for L2-normalizing descriptors.

### `global_pipeline/model/pos_embed.py`

Position-embedding utilities for 2D sine/cosine embeddings and interpolation, used by transformer-style models.

### `global_pipeline/model/r2former.py`

R2Former retrieval architecture. Defines the `R2Former` model, ResNet-50 helper `res50`, arbitrary-size patch embedding, L2 normalization, GeM pooling, and positional-resizing helpers.

## Global CCT Model Files

### `global_pipeline/model/cct/__init__.py`

Package marker/import file for CCT models.

### `global_pipeline/model/cct/cct.py`

Compact Convolutional Transformer definitions. Defines `CCT`, the internal `_cct` constructor, and many named CCT variants with different depths, kernel sizes, strides, resolutions, and positional embedding settings.

### `global_pipeline/model/cct/embedder.py`

Defines `Embedder`, a token embedding helper for CCT/transformer inputs.

### `global_pipeline/model/cct/helpers.py`

Helper functions for resizing positional embeddings and checking positional-embedding compatibility.

### `global_pipeline/model/cct/stochastic_depth.py`

Implements stochastic depth through `drop_path` and the `DropPath` module.

### `global_pipeline/model/cct/tokenizer.py`

Defines `Tokenizer` and `TextTokenizer` for converting images or token sequences into transformer-ready embeddings.

### `global_pipeline/model/cct/transformers.py`

Transformer layers and classifiers used by CCT. Defines standard and masked attention, encoder layers, and classifier heads.

## Global Pix2Pix Files

### `global_pipeline/model/pix2pix_networks/__init__.py`

Empty package marker.

### `global_pipeline/model/pix2pix_networks/LICENSE`

License for Pix2Pix-derived network code.

### `global_pipeline/model/pix2pix_networks/networks.py`

Pix2Pix network components. Defines GAN loss handling, U-Net generator blocks, PatchGAN discriminator, and scheduler construction.

## Synchronized BatchNorm Files

The same synchronized batch normalization implementation appears in:

- `global_pipeline/model/sync_batchnorm/`
- `local_pipeline/model/sync_batchnorm/`
- `keypoint_pipeline/myloftr/model/sync_batchnorm/`
- `keypoint_pipeline/myr2d2/model/sync_batchnorm/`

Each copy has the same role:

- `__init__.py`: exports synchronized batch norm and callback-enabled data parallel helpers.
- `batchnorm.py`: defines synchronized 1D/2D/3D batch norm modules, conversion helpers, and patching utilities.
- `batchnorm_reimpl.py`: standalone `BatchNorm2dReimpl` for numerical testing.
- `comm.py`: thread-safe master/slave communication primitives used during multi-GPU synchronization.
- `replicate.py`: data-parallel replication callback support.
- `unittest.py`: tensor comparison helper for tests.

## Local Pipeline Files

### `local_pipeline/LICENSE`

License file for local pipeline code.

### `local_pipeline/__init__.py`

Empty package marker.

### `local_pipeline/commons.py`

Shared logging and deterministic seeding helpers, equivalent in purpose to `global_pipeline/commons.py`.

### `local_pipeline/corr.py`

Correlation volume implementation for local alignment. Defines `CorrBlock` and `CorrBlockSingleScale`, which compute and sample feature correlations used by the iterative homography updater.

### `local_pipeline/datasets_4cor_img.py`

Dataset code for supervised four-corner homography training. Defines `homo_dataset` and `MYDATA`, plus `fetch_dataloader` and worker seeding. It loads satellite/thermal H5 data, applies crops/augmentations, and returns tensors and corner displacement targets.

### `local_pipeline/evaluate.py`

Validation helper exposing `validate_process`, used during local model training to compute evaluation metrics.

### `local_pipeline/extractor.py`

Feature extractor networks for local matching. Defines residual and bottleneck blocks plus `BasicEncoder` and `BasicEncoderQuarter`, which produce dense feature maps for correlation.

### `local_pipeline/myevaluate.py`

Main local evaluation script. Defines `test` and `evaluate_SNet`, loads trained models, evaluates homography displacement and alignment quality, and has a CLI entry point.

### `local_pipeline/parser.py`

Argument parser for local training/evaluation. It includes model size, crop size, displacement channel count, batch size, learning rate, checkpoint restore, two-stage/refinement, and dataset settings.

### `local_pipeline/plot_hist.py`

Small CLI plotting helper. Defines `plot_hist_helper` and can be run directly to visualize result distributions.

### `local_pipeline/train_4cor.py`

Main local training script. Defines `main`, `train`, and `validate`; builds dataloaders, initializes the model, runs training iterations, computes losses, evaluates, and saves checkpoints.

### `local_pipeline/update.py`

Iterative update network for homography displacement. Defines CNN variants (`CNN_weight`, `CNN_weight_64`, `CNN_128`, `CNN_64`, `CNN`) and `GMA`, which predicts updates from correlation/flow features.

### `local_pipeline/utils.py`

Local pipeline utilities. Includes bilinear sampling, coordinate-grid creation, image/overlap saving, seeding, parameter counting, optical-flow-style warping, sequence/single losses, and optimizer/scheduler construction.

## Local Model Files

### `local_pipeline/model/__init__.py`

Empty package marker.

### `local_pipeline/model/network.py`

Core trainable local STHN model. Defines `IHN` for iterative homography prediction and `STHN` for coarse/fine model setup, inputs, forward pass, cropped refinement, loss computation, optimization, and learning-rate updates. Also defines `mywarp`.

### `local_pipeline/model/pix2pix_networks/__init__.py`

Empty package marker.

### `local_pipeline/model/pix2pix_networks/LICENSE`

License for the Pix2Pix-derived local discriminator code.

### `local_pipeline/model/pix2pix_networks/networks.py`

Local Pix2Pix discriminator utilities. Defines `GANLoss` and `NLayerDiscriminator`, mainly for adversarial components.

## Keypoint Pipeline: `myloftr`

### `keypoint_pipeline/myloftr/commons.py`

Logging and deterministic seeding helpers.

### `keypoint_pipeline/myloftr/datasets_4cor_img.py`

Four-corner homography dataset implementation for the LoFTR-style keypoint pipeline. Defines `homo_dataset`, `MYDATA`, `MYTRIPLETDATA`, `fetch_dataloader`, and `seed_worker`.

### `keypoint_pipeline/myloftr/myevaluate.py`

Evaluation script for the LoFTR-style keypoint pipeline. Defines `load_model`, `test`, and `evaluate_SNet`, and has a CLI entry point.

### `keypoint_pipeline/myloftr/parser.py`

Argument parser for LoFTR-style keypoint evaluation/training settings.

### `keypoint_pipeline/myloftr/plot_hist.py`

Histogram plotting CLI helper for evaluation result distributions.

### `keypoint_pipeline/myloftr/utils.py`

Utility functions for sampling, warping, saving images, losses, and optimizer setup. It extends the local utilities with negative-pair losses such as `single_neg_loss` and `sequence_neg_loss`.

### `keypoint_pipeline/myloftr/model/network.py`

Defines `KeyNet`, a keypoint/homography network wrapper for LoFTR-style matching, plus `mywarp`.

### `keypoint_pipeline/myloftr/model/sync_batchnorm/*`

Duplicated synchronized batch normalization support described in the synchronized batch norm section.

## Keypoint Pipeline: `myr2d2`

### `keypoint_pipeline/myr2d2/commons.py`

Logging and deterministic seeding helpers.

### `keypoint_pipeline/myr2d2/datasets_4cor_img.py`

Four-corner homography dataset implementation for R2D2-style keypoint training/evaluation. Defines `homo_dataset`, `MYDATA`, `MYTRIPLETDATA`, `fetch_dataloader`, and `seed_worker`.

### `keypoint_pipeline/myr2d2/evaluate.py`

Validation helper exposing `validate_process`.

### `keypoint_pipeline/myr2d2/extract.py`

R2D2 feature extraction utilities. Defines `NonMaxSuppression`, network loading, multiscale descriptor extraction, and keypoint extraction.

### `keypoint_pipeline/myr2d2/myevaluate.py`

Main R2D2-style evaluation script. Defines `load_model`, `test`, and `evaluate_SNet`, and has a CLI entry point.

### `keypoint_pipeline/myr2d2/parser.py`

Argument parser for R2D2-style training/evaluation settings.

### `keypoint_pipeline/myr2d2/plot_hist.py`

Histogram plotting CLI helper.

### `keypoint_pipeline/myr2d2/train_key.py`

Training script for the R2D2-style keypoint network. Defines `main`, `train`, and `validate`, builds data, trains, validates, and saves checkpoints.

### `keypoint_pipeline/myr2d2/utils.py`

Utility functions for sampling, warping, saving visualizations, positive/negative sequence losses, and optimizer setup.

### `keypoint_pipeline/myr2d2/model/ap_loss.py`

Defines `APLoss`, an average-precision-style differentiable loss used in R2D2-style descriptor/reliability training.

### `keypoint_pipeline/myr2d2/model/losses.py`

Defines `MultiLoss`, a wrapper that combines repeatability, reliability, and descriptor losses.

### `keypoint_pipeline/myr2d2/model/network.py`

Defines `KeyNet`, the R2D2-style keypoint/homography network wrapper, plus `mywarp`.

### `keypoint_pipeline/myr2d2/model/patchnet.py`

Patch descriptor network definitions. Includes base patch network classes and L2-Net/Quad-L2-Net/Fast-Quad-L2-Net variants with optional confidence heads.

### `keypoint_pipeline/myr2d2/model/reliability_loss.py`

Defines `PixelAPLoss` and `ReliabilityLoss` for training confidence/reliability maps.

### `keypoint_pipeline/myr2d2/model/repeatability_loss.py`

Defines `CosimLoss` and `PeakyLoss` for repeatability and peaked response training.

### `keypoint_pipeline/myr2d2/model/sampler.py`

Sampling strategies for descriptor and keypoint training. Defines full, sub-sampled, neighborhood, far/near, and second neighborhood samplers.

### `keypoint_pipeline/myr2d2/model/sync_batchnorm/*`

Duplicated synchronized batch normalization support described earlier.

## R2D2 Tool Files

### `keypoint_pipeline/myr2d2/tools/common.py`

General R2D2 tool helpers for directory creation, model-size reporting, and GPU selection.

### `keypoint_pipeline/myr2d2/tools/dataloader.py`

Pair dataloader implementation for R2D2 training. Defines `PairLoader`, threaded loading, custom collation, image conversion, and a CLI test entry point.

### `keypoint_pipeline/myr2d2/tools/trainer.py`

Defines `Trainer`, a training harness abstraction for iterating, logging, and optimizing R2D2-style models.

### `keypoint_pipeline/myr2d2/tools/transforms.py`

Image augmentation/transformation classes for R2D2. Includes scaling, random scaling, random/center cropping, rotation, tilting, still transform, pixel noise, color jitter, and transformation instantiation.

### `keypoint_pipeline/myr2d2/tools/transforms_tools.py`

Low-level image transformation helpers. Includes image grabbing, label updates, random log-uniform sampling, translation, rotation, perspective transforms, PIL checks, and brightness/contrast/saturation/hue adjustments.

### `keypoint_pipeline/myr2d2/tools/viz.py`

Visualization helpers for optical flow. Defines color-wheel generation, flow color computation, flow-to-color conversion, and display helpers.

## Repository-Level Utility Files

### `utils/compare.py`

Short H5 comparison helper. It imports `h5py` and `tqdm` and is intended for inspecting or comparing H5 dataset contents.

### `utils/plotting.py`

Duplicate plotting helper with `process_results_simulation`, equivalent in role to `global_pipeline/plotting.py`.

## Slurm Script Structure

The `scripts/` tree contains cluster launchers. These files are not Python modules, but they are important coding/configuration files because they define experiment commands, resources, arguments, and checkpoints.

```text
scripts/
|-- global/
|   |-- eval.sbatch
|   |-- eval_anyloc.sbatch
|   |-- eval_satellite_translation_exclude_dense.sbatch
|   |-- train_bing_thermal_partial_resnet50_gem_extended.sbatch
|   |-- train_bing_thermal_partial_resnet50_gem_extended_DANN.sbatch
|   |-- train_bing_thermal_partial_resnet50_netvlad_extended.sbatch
|   |-- train_bing_thermal_partial_resnet50_netvlad_extended_DANN.sbatch
|   |-- train_bing_thermal_translation_100.sbatch
|   `-- train_bing_thermal_translation_100_nocontrast.sbatch
|-- local/
|   |-- eval_local_sparse_512_extended.sbatch
|   |-- train_local_dense.sbatch
|   |-- train_local_dense_extended.sbatch
|   |-- train_local_sparse_64.sbatch
|   |-- train_local_sparse_64_extended.sbatch
|   |-- train_local_sparse_128.sbatch
|   |-- train_local_sparse_128_extended.sbatch
|   |-- train_local_sparse_256.sbatch
|   |-- train_local_sparse_256_extended.sbatch
|   |-- train_local_sparse_512.sbatch
|   `-- train_local_sparse_512_extended.sbatch
|-- local_large/
|   |-- train_local_dense_extended.sbatch
|   |-- train_local_sparse_64_extended.sbatch
|   |-- train_local_sparse_128_extended.sbatch
|   |-- train_local_sparse_256_extended_long.sbatch
|   `-- train_local_sparse_512_extended_long.sbatch
|-- local_larger/
|   |-- eval_local_sparse_512_extended.sbatch
|   |-- train_local_dense_extended.sbatch
|   |-- train_local_sparse_64_extended.sbatch
|   |-- train_local_sparse_128_extended.sbatch
|   |-- train_local_sparse_256_extended_long.sbatch
|   `-- train_local_sparse_512_extended_long.sbatch
|-- local_larger_2/
|   |-- eval_local_sparse_512_extended.sbatch
|   |-- train_local_dense_extended_load_f_aug64_c.sbatch
|   |-- train_local_sparse_64_extended_load_f_aug64_c.sbatch
|   |-- train_local_sparse_128_extended_load_f_aug64_c.sbatch
|   |-- train_local_sparse_256_extended_long_load_f_aug64_c.sbatch
|   `-- train_local_sparse_512_extended_long_load_f_aug64_c.sbatch
`-- local_larger_augment/
    |-- eval_local_sparse_512_extended.sbatch
    |-- eval_local_sparse_512_extended_2.sbatch
    |-- train_local_sparse_512_extended_long.sbatch
    `-- train_local_sparse_512_extended_long_load_f_aug64_c.sbatch
```

### Slurm Script Naming Pattern

- `global/eval*.sbatch`: evaluates global retrieval or AnyLoc models.
- `global/train_bing_thermal_partial_resnet50_*`: trains ResNet-50 global retrieval models using GeM or NetVLAD, optionally with DANN.
- `global/train_bing_thermal_translation_*`: trains thermal generation/Pix2Pix models.
- `local/train_local_dense*`: trains dense local homography variants.
- `local/train_local_sparse_*`: trains sparse local variants with displacement channel sizes such as 64, 128, 256, or 512.
- `local_large/` and `local_larger/`: larger-resolution versions of local experiments.
- `local_larger_2/*load_f_aug64_c*`: two-stage/refinement experiments that load a previous coarse model and use augmentation/crop settings.
- `local_larger_augment/`: augmentation-focused larger-resolution local experiments.

## Non-Source and Generated Files

Several `__pycache__/` folders contain compiled `.pyc` files. They are runtime artifacts generated by Python and should not be edited manually. They are not documented individually because their source `.py` files are documented above.

## Important Entry Points

- Demo inference: `python STHN_demo.py`
- Two-stage demo inference: `python STHN_demo.py --two_stages`
- Local training: `local_pipeline/train_4cor.py`
- Local evaluation: `local_pipeline/myevaluate.py`
- Global retrieval training: `global_pipeline/train.py`
- Global retrieval evaluation: `global_pipeline/eval.py`
- AnyLoc evaluation: `global_pipeline/eval_anyloc.py`
- Pix2Pix/TGM training: `global_pipeline/train_pix2pix.py`
- Dataset H5 creation: `global_pipeline/h5_transformer.py`
- Dataset H5 merging: `global_pipeline/h5_merger.py`
- R2D2-style keypoint training: `keypoint_pipeline/myr2d2/train_key.py`

## Notes for Future Maintainers

- The codebase contains multiple duplicated helper modules. When fixing bugs in common utilities, check whether the same file exists in global, local, LoFTR, and R2D2 folders.
- `global_pipeline/eval_pix2pix.py` should be syntax-checked before use because AST parsing found an indentation error.
- The repository uses old-style relative imports in many places. Running modules from the wrong directory may cause import errors.
- The Slurm scripts encode important experiment settings. If reproducing paper results, inspect the exact `.sbatch` file rather than relying only on Python defaults.
- Dataset files are external and large; most loaders expect H5 files with image arrays and metadata matching the README-described folder structure.
