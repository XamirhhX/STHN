# STHN Algorithm Explanation

This document explains the algorithmic idea behind how the STHN model finds matching parts between a satellite image and a thermal UAV image, then estimates the homography that aligns them.

The important point is that the model does **not** recognize semantic object classes such as "building", "road", "tree", or "vehicle" explicitly. Instead, it learns dense visual features for small image regions, compares those regions across the satellite and thermal images, and predicts how the four corners of one image must move so both images geometrically align.

## Problem Being Solved

The model receives two images:

- A satellite RGB image.
- A UAV thermal image.

The images may show the same physical area, but from different sensors and possibly different viewpoints. The goal is to estimate a homography, which is a 3x3 perspective transformation matrix that can warp one image into the coordinate frame of the other.

Instead of directly predicting the 3x3 matrix, STHN predicts a **four-point displacement**:

```text
top-left corner displacement
top-right corner displacement
bottom-left corner displacement
bottom-right corner displacement
```

These four moved corners define the homography. The code uses Kornia's perspective transform utilities to convert the predicted corner motion into a warp.

## Main Algorithm in One Sentence

STHN converts both images into dense feature maps, builds a correlation volume that measures how well each local feature patch matches nearby patches in the other image, repeatedly refines a four-corner displacement prediction, and finally converts that displacement into a homography.

## Main Components

The core local alignment algorithm is implemented in these files:

- `local_pipeline/model/network.py`: defines the `IHN` and `STHN` model wrappers.
- `local_pipeline/extractor.py`: defines the CNN feature extractor.
- `local_pipeline/corr.py`: defines the correlation volume.
- `local_pipeline/update.py`: defines the CNN update block that predicts corner-displacement corrections.
- `local_pipeline/utils.py`: defines sampling, warping, losses, and optimizer helpers.

The standalone demo reimplements the needed inference parts in:

- `STHN_demo.py`

## Stage 1: Image Preprocessing

Before matching, both images are resized and normalized.

In `local_pipeline/model/network.py`, the `IHN.forward` method normalizes images using ImageNet statistics:

```text
normalized_image = (image - ImageNet_mean) / ImageNet_std
```

This makes the input distribution more stable for the CNN feature extractor. The thermal image is handled as a 3-channel tensor in the model path, so the same feature extractor can process both satellite and thermal inputs.

## Stage 2: Dense Feature Extraction

The model does not compare raw pixels directly. Satellite and thermal images look very different at the pixel level, so raw RGB/thermal intensity comparison would be unreliable.

Instead, `BasicEncoderQuarter` in `local_pipeline/extractor.py` converts each image into a dense feature map.

For an input image of size `W x W`, the encoder produces a feature map at roughly `W/4 x W/4` spatial resolution. Each cell in this feature map represents a local patch of the original image.

Conceptually:

```text
satellite image -> CNN encoder -> satellite feature map
thermal image   -> CNN encoder -> thermal feature map
```

Each feature vector encodes local visual structure such as:

- edges,
- corners,
- texture transitions,
- road/building boundaries,
- shape patterns,
- thermal/satellite correspondences learned during training.

These are not hand-coded rules. They are learned from training examples where the correct homography displacement is known.

## How the Model "Recognizes Parts"

The model recognizes parts through **learned local descriptors**, not semantic labels.

A "part" is effectively a feature-map cell or a neighborhood of cells. The CNN learns to map visually corresponding satellite and thermal regions into feature vectors that can be compared.

For example, if a road intersection appears in both images, the raw pixels may look different, but the trained encoder can produce feature vectors that are more similar than unrelated regions. The model then uses correlation scores to detect that those two locations likely correspond.

So the process is:

```text
image patch -> learned descriptor -> descriptor similarity -> possible match
```

The model does not say:

```text
this is a road
this is a roof
this is a field
```

It instead learns:

```text
this local pattern in image A is compatible with this local pattern in image B
```

## Stage 3: Correlation Volume Construction

The correlation volume is built in `local_pipeline/corr.py`.

Given two feature maps:

```text
fmap1: satellite features
fmap2: thermal features
```

The model computes dot-product similarity between feature vectors:

```text
similarity(i, j) = feature_from_image_1(i) dot feature_from_image_2(j)
```

This creates a large matching table:

```text
for every location in image 1:
    compare it with locations in image 2
```

In code, this happens in `CorrBlock.corr`:

```text
corr = fmap1^T x fmap2
```

The output says how strongly each location in the satellite feature map matches each location in the thermal feature map.

## Stage 4: Correlation Pyramid

The model does not only match at one scale. `CorrBlock` builds a small pyramid by average-pooling the correlation map.

This gives the updater access to:

- fine local matching evidence,
- broader lower-resolution matching evidence,
- more tolerance to initial misalignment.

This is useful because satellite and thermal images can be shifted, rotated by perspective, or misaligned by a large amount.

## Stage 5: Initialize the Coordinate Grid

The model starts from an identity alignment.

In `IHN.initialize_flow_4`, two coordinate grids are created:

```text
coords0 = original feature-grid coordinates
coords1 = current estimated matched coordinates
```

Initially:

```text
coords1 == coords0
```

That means the model starts by assuming no warp. It then iteratively updates the four-corner displacement.

## Stage 6: Iterative Homography Refinement

The central loop is in `IHN.forward`.

At each iteration:

1. Sample correlation evidence around the current coordinate estimate.
2. Compute the current flow:

```text
flow = coords1 - coords0
```

3. Concatenate correlation evidence and flow.
4. Pass that tensor into the update network `GMA`.
5. Predict a small correction to the four-corner displacement.
6. Add the correction to the running displacement estimate.
7. Convert the four-corner displacement back into a full coordinate flow using a perspective transform.

Conceptually:

```text
four_point_disp = 0

for each refinement iteration:
    local_matching_scores = correlation(current_alignment)
    current_flow = current_coordinates - original_coordinates
    delta = update_network(local_matching_scores, current_flow)
    four_point_disp = four_point_disp + delta
    current_alignment = homography_from(four_point_disp)
```

This is similar in spirit to optical-flow refinement models: rather than predicting the final alignment in one step, the network repeatedly corrects its estimate.

## Stage 7: The Update Network

The update block is `GMA` in `local_pipeline/update.py`.

Despite the name, this implementation uses CNN blocks to read the correlation/flow tensor and output a two-channel corner displacement correction.

Input to the updater:

```text
[correlation features, current flow]
```

Output from the updater:

```text
delta_four_point
```

The output is added to the current four-corner displacement. Different CNN variants are selected depending on feature-map size and settings:

- `CNN`
- `CNN_64`
- `CNN_128`
- `CNN_weight`
- `CNN_weight_64`

The weighted variants also predict a spatial weight map. This lets the model emphasize more reliable matching regions and suppress weak or ambiguous regions.

## Stage 8: Four-Corner Displacement to Homography

The model predicts displacement for the four image corners:

```text
original corners:
    (0, 0)
    (W - 1, 0)
    (0, W - 1)
    (W - 1, W - 1)

predicted corners:
    original corners + predicted displacement
```

In `IHN.get_flow_now_4`, the code calls:

```text
kornia.geometry.transform.get_perspective_transform
```

This computes a homography from the original four corners to the predicted four corners. The homography is then applied to every coordinate in the feature grid to update `coords1`.

At visualization or evaluation time, `mywarp` uses:

```text
kornia.geometry.transform.warp_perspective
```

to warp the image according to the predicted homography.

## Stage 9: Training Signal

During training, the dataset provides the expected geometric displacement. The model compares predicted displacement against ground truth.

The local pipeline uses losses from `local_pipeline/utils.py`, especially:

- `sequence_loss`
- `single_loss`

The sequence loss supervises multiple refinement iterations. This encourages the model to improve progressively, not only at the final iteration.

The training target is geometric:

```text
predicted corner displacement should match ground-truth corner displacement
```

Because the supervision is geometric, the model learns whatever visual evidence helps predict alignment. It is not trained as a semantic segmentation or object recognition model.

## One-Stage STHN

In one-stage mode:

```text
satellite crop + thermal image
        |
        v
IHN predicts four-corner displacement
        |
        v
homography is computed
        |
        v
thermal/satellite alignment is evaluated or visualized
```

This is the simpler path used when `--two_stages` is not enabled.

## Two-Stage STHN

Two-stage mode adds a refinement step.

First, a coarse model estimates an approximate alignment. Then the model crops the satellite image around the predicted matched region and runs a fine model on that crop.

The two-stage process:

```text
1. Run coarse IHN on resized satellite image and thermal image.
2. Convert coarse four-corner prediction into a bounding box.
3. Crop the original larger satellite image around that predicted box.
4. Resize the crop to the model input size.
5. Run fine IHN on the cropped satellite image and thermal image.
6. Convert the fine prediction back to the original coordinate system.
7. Combine coarse and fine predictions.
```

The crop logic is implemented in:

```text
STHN.get_cropped_st_images
```

The prediction-combination logic is implemented in:

```text
STHN.combine_coarse_fine
```

The motivation is that a large satellite image can be difficult to align precisely in one pass. The coarse stage finds the approximate region, and the fine stage focuses on local details inside that region.

## Why Correlation Helps Cross-Modal Matching

Satellite RGB and thermal images have different appearances. A direct pixel comparison fails because the same physical surface can have very different values in the two modalities.

The trained encoder learns a representation where corresponding structures become easier to compare. Correlation then turns this representation into explicit matching evidence.

Useful cross-modal cues can include:

- object boundaries,
- road layout,
- building footprints,
- large shape contours,
- spatial arrangement of high-contrast regions,
- repeated map patterns,
- thermal edges corresponding to physical structures.

The model learns these cues from data, not from manually specified rules.

## What the Model Is Actually Predicting

The final output is not a class label and not a set of named objects. It is a geometric transformation.

The direct prediction is:

```text
four_point_displacement: Tensor[B, 2, 2, 2]
```

Meaning:

```text
B     = batch size
2     = x/y displacement channels
2 x 2 = four image corners arranged as a grid
```

This displacement is enough to compute:

```text
homography matrix H: Tensor[B, 3, 3]
```

The homography is then used to warp one image into the other.

## Algorithm Pseudocode

```text
function STHN_ALIGN(satellite_image, thermal_image):
    satellite = resize_and_normalize(satellite_image)
    thermal = resize_and_normalize(thermal_image)

    satellite_features = encoder(satellite)
    thermal_features = encoder(thermal)

    corr_pyramid = build_correlation_pyramid(
        satellite_features,
        thermal_features
    )

    coords0 = regular_grid()
    coords1 = regular_grid()
    four_point_disp = zeros()

    for iteration in range(num_iterations):
        corr_features = sample_correlation(corr_pyramid, coords1)
        flow = coords1 - coords0

        delta = update_network(corr_features, flow)
        four_point_disp = four_point_disp + delta

        H = perspective_transform(
            original_corners,
            original_corners + four_point_disp
        )

        coords1 = apply_homography_to_grid(H)

    return four_point_disp, H
```

## Difference Between Global and Local Algorithms

The repository also contains a global pipeline. Its role is different from the local STHN homography network.

### Global Pipeline

The global pipeline asks:

```text
Which satellite database image is most likely to correspond to this thermal query?
```

It uses retrieval-style descriptors and aggregation methods such as GeM, NetVLAD, or AnyLoc/VLAD.

### Local Pipeline

The local pipeline asks:

```text
Given this satellite image and thermal image, what homography aligns them?
```

It uses dense feature correlation and iterative four-corner displacement prediction.

In a full localization system, the global stage can narrow the search area, while the local stage performs precise geometric alignment.

## Why the Model Can Align Without Semantic Labels

A homography can be estimated from repeated local geometric evidence. The model only needs enough corresponding visual structures to determine how the image plane moved.

For example, if many feature patches agree that:

```text
top-left region should move slightly right
top-right region should move downward
bottom corners should expand outward
```

then the update network can infer the corner displacement that best explains those matches.

This is why explicit part names are unnecessary. The "parts" are learned feature regions, and the final decision is geometric consistency across many regions.

## Failure Cases

The algorithm can struggle when:

- The satellite and thermal images do not overlap enough.
- The region has few distinctive structures.
- The thermal image is noisy or low contrast.
- The initial crop is too far from the true location.
- Repeated patterns create ambiguous matches.
- The true transformation is not well approximated by a homography.
- The predicted four-corner configuration becomes geometrically invalid.

The code has defensive handling for some invalid transforms. For example, if a perspective transform cannot be solved, it may ignore a bad update or fall back to an identity transform in warping.

## Summary

STHN recognizes corresponding parts by learning dense patch descriptors and comparing them through a correlation volume. It does not assign semantic names to image regions. Instead, it uses repeated local matching evidence to iteratively predict how the four corners of the image should move. Those four moved corners define a homography, and the homography aligns the satellite and thermal images.

