# Maintained configuration files

| File | Use |
|---|---|
| `continuous_odf_sputter_pvd_broad.json` | Broad-likelihood continuous ODF corpus for sputter/PVD films. It covers FCC, BCC, and HCP PVD texture families with an explicit uniform escape channel. |
| `curated_sputter_validation.json` | Short, interpretable FCC Cu PVD validation sweep for film-normal fiber perturbations. |
| `fcc_texture_grain_example.json` | Direct single-simulation example for FCC grain and texture settings. |

Dataset products must be written below `datasets/`; that directory is ignored
by Git. Deprecated rolling, generic-Sobol, proof, and pilot experiment presets
were removed because they did not describe the maintained PVD inference path.
