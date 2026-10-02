# Prospective extension of the same SDSS compact-remnant search

Frozen on 2026-10-02, after the 44-star pilot and before downloading the
additional targets' ZTF light curves. The pilot is preserved separately.

The pilot selected the extreme edge of a compact, blue, nearby white-dwarf
population. Extend to all 400 SDSS DR20 SnowWhite sources satisfying the
unchanged base cuts: parallax/error > 10, G < 18.5, BP-RP < 0.1,
absolute G > 12 (inverse-parallax ranking only), declination > -30 degrees.
Remove only the pilot's absolute-G-minus-three-colour >=13 ranking cut.
This is a broader search within the same physical population and primary
dataset, not a claim that these are 400 massive or previously unknown stars.

Verify exact Gaia IDs against the canonical FITS file; require RUWE <1.4
and IPD multiple peak fraction <=2 before acquiring photometry. Preserve
excluded rows and metadata. Identify targets whose light curves were
examined earlier in this project. Reuse identical pilot results without
rerunning them or treating them as new evidence.

Use the unchanged version-2 period engine: exposure-midpoint BJD TDB,
exposure-integrated sinusoid, separate g/r amplitudes and phases, nuisance
image-quality regressors, constant-model extra scatter, earliest 75% of
nights for discovery, last 25% for validation. Search 0.01–1440/day,
five grid samples per Fourier spacing, freeze 20 exact-refined frequencies
before examining validation. The primary check is the discovery maximum;
other aliases remain clearly identified as multiple tests.

Use 400 x 20 as the screening family multiplier, including pilot, controls,
and excluded targets conservatively. Nominal chi-square probabilities are
not discovery probabilities. Any retained lead needs time-correlated nulls,
alias tests, contamination checks, independent observations when available,
and full primary-paper checks before novelty is claimed.

Measured pilot cost: 35 usable searches took about 20 CPU minutes in total.
The extension is expected to take a few CPU hours, at most about 1 GB of
additional public light curves, and no paid compute. Use two bounded local
workers, three download threads, immutable raw files, per-star checkpoints,
and cached scans. Continue evaluating the two pilot leads while the
extension runs. The pilot's incomplete independent confirmations are not
grounds for declaring those leads discoveries or silently dropping them.
