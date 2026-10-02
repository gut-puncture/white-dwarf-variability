# Fixed spectral feature experiment — 2026-10-02

This is an exploratory follow-up of the already selected117.406-minute ZTF/TESS candidate. Its continuum-colour measurements have been inspected. No exposure-level local-feature strengths for this star have yet been measured.

To avoid choosing a favourable absorption feature, measure every100-Angstrom-centred local equivalent-width window from3800through6800Angstrom, inclusive (31windows). Each line interval is centre±40A; continuum intervals are centre−100to−60A and centre+60to+100A. These are phenomenological shape measurements, not atomic identifications. Some windows overlap continuum intervals of others and are correlated.

Use exactly the existing strict pixel masks, at least90percent line coverage and80percent continuum coverage, with propagated local linear-continuum covariance. Fit separate visit offsets and0.5A uncertainty floor. Retain the pre-existing chronological split: first16of24nights for feature selection; last8for validation. Rank all31windows×20ZTF-discovery-frozen frequencies using training spectra alone. Save the selected feature/frequency before evaluating the validation strengths. Do not optimize spectral frequency or adjust windows after seeing results. Preserve the primary ZTF-frequency results for all windows as exploratory outputs, with explicit620test accounting.

Controls are the previously metadata-selected125comparison spectra. Any selected local-feature signal must survive the same-exposure comparison test, additional continuum/error-floor sensitivities, and a whole-night null before receiving physical interpretation. This experiment follows a continuum experiment and is not an independent blind discovery search.
