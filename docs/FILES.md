# Start with the result; follow any number back to its source

| You want to see… | Open… |
| --- | --- |
| What was found | [README](../README.md) |
| Your freshly calculated result | `results/SUMMARY.md` |
| Every checked number and tolerance | `results/verification.json` |
| Machine-readable results | `results/measurements.json` |
| Fresh plots and complete stage output | `results/figures/`, `results/logs/` |
| Original file, archive URL/query and SHA-256 | [raw_manifest.json](../provenance/raw_manifest.json) |
| Original data | The versioned release archive; unpacked under `results/raw/` |
| All 400 screening targets | [parent list](../provenance/fast_rotation_extension_parent.csv) |
| Earlier results used only for comparison | [reference/data/](../reference/data/) |

## Code order

`reproduce.py` creates an environment, verifies inputs, downloads and verifies the original observations, then creates a new output folder. Existing results are never overwritten. `run_analysis.py` runs these stages in order:

| Step | Code in `scripts/` |
| --- | --- |
| ZTF association, cleaning, time conversion, discovery and validation | `fast_rotation_search.py` |
| Image-quality regressors used by that model | `74835110_photometric_audit.py`, function `nuisance` |
| ZTF robustness, aliases, seasonal checks and night-sign null | `fast_rotation_candidate_ztf_audit.py` |
| TESS sector 72 light curve | `fast_rotation_tess_validation.py` |
| Sector 72 source location | `109631850_pixel_and_spectrum_checks.py`, function `pixels`, with target set to 80364427 |
| Sector 72 block-sign null | `80364427_tess_correlated_null.py` |
| Four further TESS sectors | `80364427_additional_tess_validation.py`, using `61247086_tesscut_validation.py` for extraction and fits |
| Four-sector pixels and block-sign nulls | `80364427_additional_tess_audit.py` |
| Unsuccessful broad-colour test; fixed-window measurements | `80364427_spectral_colours.py`, `80364427_continuum_windows.py` |
| Spectral feature selection and later-night prediction | `80364427_spectral_feature_scan.py` |
| Same-exposure comparison stars; changed extraction assumptions | `80364427_feature_controls.py`, `80364427_feature_sensitivity.py` |
| Second-star image extraction, attribution and injection checks | `65258778_ztf_image_photometry.py`, `65258778_ztf_image_period.py` |

Other scripts supply shared functions and constants. Their original numeric names were preserved to avoid disguising their origin. [Original script checksums](../provenance/original_scripts.json) let readers check that these analysis routines match the research copies.

`inputs/` holds observing plans and the previously selected exploratory secondary-frequency record needed by the original TESS audit. That record contains earlier secondary fits, but only its frequency list is used. The main ZTF frequencies and spectral feature are selected again from the early observations on every run. Secondary-frequency tests are exploratory and do not support a second-clock claim.

`verify_results.py` runs last. It compares new calculations with `reference/`, with exact counts/labels and small numerical tolerances for floating-point differences. A mismatch gives a nonzero exit status and is written to `verification.json`. Agreement is a software reproduction check, not independent scientific validation.

## Troubleshooting

- **Python missing:** install Python 3.11 or newer from [python.org](https://www.python.org/downloads/).
- **Existing `results` folder:** choose `--output results-2`. Keep the old folder for comparison.
- **Download interrupted:** rerun. The incomplete archive is replaced; successful cached downloads are hash-checked and reused.
- **Checksum failure:** stop and inspect the named file. Do not edit a checksum to make it pass.
- **Calculation failure:** open the numbered log named in the terminal. The runner does not print a success claim.
- **Offline use:** after dependencies and the archive are downloaded, `--data-archive /path/to/raw-observations-v1.tar.gz` uses that exact archive. The analysis itself makes no network requests.

Default outputs include copies of the raw observations, so each run uses additional disk space. Delete only your own generated results when you no longer need them. The raw cache and research reference remain separate.
