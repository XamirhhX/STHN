$ErrorActionPreference = "Stop"
$venv = "C:\Users\Reacher\Documents\University\Project\Project\STHN-main\STHN-main\.venv_retrieval_cpu\Scripts\python.exe"
Set-Location "C:\Users\Reacher\Documents\University\Project\Project\STHN-main\STHN-main"

& $venv scripts\professor_retrieval_benchmark.py `
  --dataset_label san_francisco_two_grid_all_queries_cpu_20260622 `
  --database_h5 "C:\Users\Reacher\Documents\University\Project\Project\Retrieval_Test_Bundle_2026-06-15\datasets\retrieval_san_francisco_two_grid\test_database.h5" `
  --queries_h5 "C:\Users\Reacher\Documents\University\Project\Project\Retrieval_Test_Bundle_2026-06-15\datasets\retrieval_san_francisco_two_grid\train_queries.h5" "C:\Users\Reacher\Documents\University\Project\Project\Retrieval_Test_Bundle_2026-06-15\datasets\retrieval_san_francisco_two_grid\val_queries.h5" "C:\Users\Reacher\Documents\University\Project\Project\Retrieval_Test_Bundle_2026-06-15\datasets\retrieval_san_francisco_two_grid\test_queries.h5" `
  --map_path "C:\Users\Reacher\Documents\University\Project\Project\Retrieval_Test_Bundle_2026-06-15\datasets\maps\satellite\20201117_BingSatellite.png" `
  --metadata_json "C:\Users\Reacher\Documents\University\Project\Project\Retrieval_Test_Bundle_2026-06-15\source_geotiff\Metadata\geotiff_metadata.json" `
  --output_dir "C:\Users\Reacher\Documents\University\Project\Project\04_Results\Retrieval\san_francisco_urban_allqueries_cpu_20260622" `
  --terrain urban `
  --location "San Francisco, CA" `
  --tile_size_px 512 `
  --device cpu `
  --model resnet18 `
  --weights imagenet `
  --recall_values 1 3 5 `
  --spiral_grid_step_m 625 `
  --spiral_search_radius_m 5000 `
  --handoff_top_n 3 `
  --handoff_neighbors 4

& $venv scripts\professor_retrieval_benchmark.py `
  --dataset_label grand_canyon_all_patches_cpu_20260622 `
  --database_h5 "C:\Users\Reacher\Documents\University\Project\Project\04_Results\STHN\preset-us-grand-canyon-model-demo-20260614-144845\STHN_Model_Input\sthn_dataset\satellite_0_thermalmapping_135_train\train_database.h5" "C:\Users\Reacher\Documents\University\Project\Project\04_Results\STHN\preset-us-grand-canyon-model-demo-20260614-144845\STHN_Model_Input\sthn_dataset\satellite_0_thermalmapping_135_train\val_database.h5" "C:\Users\Reacher\Documents\University\Project\Project\04_Results\STHN\preset-us-grand-canyon-model-demo-20260614-144845\STHN_Model_Input\sthn_dataset\satellite_0_thermalmapping_135_train\test_database.h5" `
  --queries_h5 "C:\Users\Reacher\Documents\University\Project\Project\04_Results\STHN\preset-us-grand-canyon-model-demo-20260614-144845\STHN_Model_Input\sthn_dataset\satellite_0_thermalmapping_135_train\train_queries.h5" "C:\Users\Reacher\Documents\University\Project\Project\04_Results\STHN\preset-us-grand-canyon-model-demo-20260614-144845\STHN_Model_Input\sthn_dataset\satellite_0_thermalmapping_135_train\val_queries.h5" "C:\Users\Reacher\Documents\University\Project\Project\04_Results\STHN\preset-us-grand-canyon-model-demo-20260614-144845\STHN_Model_Input\sthn_dataset\satellite_0_thermalmapping_135_train\test_queries.h5" `
  --metadata_json "C:\Users\Reacher\Documents\University\Project\Project\04_Results\STHN\preset-us-grand-canyon-model-demo-20260614-144845\Metadata\geotiff_metadata.json" `
  --output_dir "C:\Users\Reacher\Documents\University\Project\Project\04_Results\Retrieval\grand_canyon_mountains_allqueries_cpu_20260622" `
  --terrain mountains `
  --location "Grand Canyon, AZ" `
  --tile_size_px 512 `
  --device cpu `
  --model resnet18 `
  --weights imagenet `
  --recall_values 1 3 5 `
  --spiral_search_radius_m 5000 `
  --handoff_top_n 3 `
  --handoff_neighbors 4

& $venv scripts\professor_retrieval_benchmark.py `
  --dataset_label iran_lut_kaluts_50km_two_grid_cpu_20260622 `
  --database_h5 "C:\Users\Reacher\Documents\University\Project\Project\Retrieval_Professor_Scale_Bundle_2026-06-15\datasets\retrieval_iran_lut_kaluts_50km_two_grid\test_database.h5" `
  --queries_h5 "C:\Users\Reacher\Documents\University\Project\Project\Retrieval_Professor_Scale_Bundle_2026-06-15\datasets\retrieval_iran_lut_kaluts_50km_two_grid\test_queries.h5" `
  --output_dir "C:\Users\Reacher\Documents\University\Project\Project\04_Results\Retrieval\iran_lut_kaluts_professor_scale_cpu_20260622" `
  --terrain "desert with features" `
  --location "Lut Desert / Kaluts, Iran" `
  --tile_size_px 512 `
  --device cpu `
  --model resnet18 `
  --weights imagenet `
  --recall_values 1 3 5 `
  --spiral_grid_step_m 2000 `
  --spiral_search_radius_m 50000 `
  --handoff_top_n 3 `
  --handoff_neighbors 4
