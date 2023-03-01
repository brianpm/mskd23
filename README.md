[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7689511.svg)](https://doi.org/10.5281/zenodo.7689511)

# mskd23
Code supporting manuscript.


## Notebooks for Figures
- Figures 1 & 2: `compare_ceres.ipynb`
- Figures 3 & 4: `panel_of_maps_from_catalogs.ipynb`
- Figure 5: `zonal_mean_cloud_from_catalogs.ipynb`
- Figure 6: `combined_histogram_error_plot.ipynb`
- Figures 7, 8, 9: `comparison_regimes.ipynb`
- Figure 10: `combined_histogram_error_by_omega_plot.ipynb`

## Ancillary Code
- `computations.py`: Various utility functions including averaging, masking, time alignment, weighting, binned statistics, NMSE, EMD, COSP-specific layer cloud definitions, and dataclasses used in some notebooks.
-  `compute_save_emd.py`: computes and saves EMD (using temporal overlap)
- `emd_from_climo.py`: computes and saves EMD (using climatology)
- `e3sm_derived_variable_registry.py`
- `CAM_derived_variable_registry.py`
- `plotting_methods.py`: basic formatting for dynamical regimes line plots, saving figures, Regions dataclass
- `build_intake_catalog.ipynb`: Notebook used to build the intake-esm catalogs used by some of the analysis notebooks
