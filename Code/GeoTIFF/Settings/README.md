# Preset Settings

Use these presets with `Run.bat` from the app folder:

```powershell
cd Code\GeoTIFF
.\Run.bat --settings Settings\Fast_US_GeoTIFF_Only.json
```

Preview a preset without downloading tiles:

```powershell
.\Run.bat --settings Settings\US_San_Francisco_Model_Demo.json --dry-run
```

Every preset has `"unique_folder": true`, so rerunning it creates a timestamped result folder instead of overwriting an existing result.

## Presets

- `Fast_US_GeoTIFF_Only.json` - small San Francisco GeoTIFF-only smoke test.
- `US_San_Francisco_Model_Demo.json` - dense urban US model-input demo.
- `US_New_York_Model_Demo.json` - dense grid/urban US model-input demo.
- `US_Grand_Canyon_Model_Demo.json` - US natural terrain model-input demo.
- `Iran_Lut_Kaluts_Model_Demo.json` - current Iran desert preset.
