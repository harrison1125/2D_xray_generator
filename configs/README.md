# Maintained configuration files

| File | Use |
|---|---|
| `balanced_symmetry_odf_pf_v1.json` | Matched 25k-per-class FCC/BCC/HCP benchmark with exact ODF and realized PF/IPF targets; CLI overrides support disk-limited subsets. |
| `continuous_odf_sputter_pvd_broad.json` | Broad-likelihood continuous ODF corpus for sputter/PVD films. It covers FCC, BCC, and HCP PVD texture families with an explicit uniform escape channel. |
| `curated_sputter_validation.json` | Short, interpretable FCC Cu PVD validation sweep for film-normal fiber perturbations. |
| `fcc_texture_grain_example.json` | Direct single-simulation example for FCC grain and texture settings. |

Dataset products must be written below `datasets/`; that directory is ignored
by Git. Deprecated rolling, proof, and pilot experiment presets were removed;
the balanced generic-Sobol benchmark is retained specifically for controlled
cross-symmetry inference experiments.
