# STHN Super Short Memorization Guide

## Main Idea

STHN aligns a satellite image with a thermal image by matching learned local features and predicting how the four image corners should move.

## Memorize: F-C-U-H

```text
Features -> Correlation -> Update -> Homography
```

## Meaning

- **Features**: CNN extracts useful patch descriptors from both images.
- **Correlation**: matching patches get high similarity scores.
- **Update**: the model repeatedly corrects the four corner positions.
- **Homography**: final corners create a 3x3 warp matrix.

## One Sentence Answer

STHN extracts dense features from satellite and thermal images, correlates local patches to find matches, iteratively updates four-corner displacement, and converts those corners into a homography for alignment.

## Key Point

It does not recognize named objects. It recognizes matching visual patterns.
