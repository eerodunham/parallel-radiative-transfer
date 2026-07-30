# parallel-radiative-transfer
Port of the Barrow Group's radiative transfer solver to CUDA/C++.

## Exporting Data
To export halo data into .npy files, use the exporter script:
```bash
python src/python/export_data.py [path/to/data/config] [path/to/dump/files] --timestep [timestep] --halo-version [year] --test [boolean]
```   
"Test" toggles whether or not to skip loading hyperion dust data.
Timestep defaults to 0, year defaults to 2020, and test defaults to true. 
Timestep data is not included in this repository! It is too big and should be downloaded locally.
I may add a small test region to validate builds.