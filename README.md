# STHN Project

Clean upload-ready repo for the STHN codebase and the newer GeoTIFF/retrieval work.

## Layout

- `Code/STHN` - original STHN source plus the newer scripts and documentation that were added later.
- `Code/GeoTIFF` - GeoTIFF tooling, presets, and configuration code.
- `Code/Retrieval` - standalone retrieval prototype scripts, run guide, environment file, and small JSON settings/metadata.
- `Docs/Notes` - project notes copied from the working folders.
- `References/STHN_Summaries` - lightweight STHN summary notes.
- `References/Paper_Extracts` - small text extracts kept for project context.
- `Artifacts` - local-only outputs. This folder is ignored by Git except for its README.

## What Was Left Out

The previous project folders included many generated or bulky files that should not be uploaded with source code:

- virtual environments
- caches and `__pycache__`
- generated test/result folders
- datasets, tiles, H5 files, TIFFs, PNG previews, and extracted resources
- archive zips
- large PDFs, Word files, videos, and spreadsheet binaries
- duplicate nested project copies

Those files were not deleted from the old folders. They were simply not copied into this clean repo.

## Setup

STHN dependencies are in `Code/STHN/env.yml`.

```powershell
conda env create -f Code/STHN/env.yml
```

GeoTIFF dependencies are in `Code/GeoTIFF/requirements.txt`.

```powershell
python -m pip install -r Code/GeoTIFF/requirements.txt
```
