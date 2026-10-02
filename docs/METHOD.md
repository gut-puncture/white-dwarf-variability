# What the evidence supports

The first star has a repeatable 117.4-minute brightness cycle, with spectral changes at the same frequency. The second has a 3.431-day candidate brightness cycle. These are measurements of known stars. Their physical causes remain unresolved.

The default command rebuilds the numerical evidence for these two statements from original archive files. It repeats the period searches, quality cuts, independent-observation checks, image extraction, comparison-star tests and specified noise simulations. It also repeats an unsuccessful broad spectral-colour test.

## Why these stars were examined

The starting survey was SDSS DR20's SnowWhite white-dwarf catalogue. The extension contained 400 stars selected with parallax/error > 10, Gaia G < 18.5, BP−RP < 0.1, absolute G > 12, and declination > −30°. Gaia quality cuts required RUWE < 1.4 and multiple-peak fraction ≤ 2. These cuts select a compact-star sample; they do not measure stellar masses.

The [original plan](../provenance/fast_rotation_extension_plan.md), [400-star parent list](../provenance/fast_rotation_extension_parent.csv), [quality decisions](../inputs/data/fast_rotation_extension_target_plan.csv), and [screening results](../provenance/fast_rotation_extension_final_primary_screen.csv) preserve the wider search. The reproduction reruns these two targets, not the entire 400-star search or its initial catalogue extraction. The screening family is still counted as 400, not reduced to two after selection.

The plans are preserved research records. Their dates are not an external preregistration. Spectral follow-up began after some spectra and continuum behaviour had been inspected; it was not a wholly blind experiment.

## Find the periods in ZTF

The code associates each ZTF object with the Gaia proper-motion track, within 1.5 arcseconds. It keeps g/r measurements with zero catalogue flags, positive exposure time and magnitude error between 0 and 0.15 mag. Duplicate exposure/filter records are removed, retaining the smallest reported error. No measurement is clipped because of its brightness or phase.

Times become exposure-midpoint BJD TDB minus 2,458,000 days, using the Palomar location. Whole nights are ordered, with the earliest 75% assigned to discovery and the rest to validation. This yields 633/126 measurements for the first star and 1,437/378 for the second.

The model contains separate offsets for ZTF objects, seven image-quality terms per band, a 0.005-mag error floor, and extra scatter fitted under the nonperiodic model. A sinusoid has separate g/r amplitudes and phases. Finite exposure time attenuates the sinusoid by `sinc(frequency × exposure_days)`.

The search spans 0.01–1,440 cycles/day, at five samples per Fourier spacing. It scans about 12.6 million frequencies for the first star and 13.5 million for the second. Leading candidates are refined using discovery data only. The best 20 are saved before validation is fitted.

| Measurement | SDSS J112148.80+103934.2 | WDJ215009.13+471257.26 |
| --- | ---: | ---: |
| Discovery period, rounded | 7,044.361 seconds | 296,469.373 seconds |
| Discovery fit improvement | 231.83 | 149.99 |
| Later-data fit improvement | 33.57 | 53.04 |
| Later improvement with earlier amplitude/phase unchanged | 25.58 | 33.75 |

“Fit improvement” means the reduction in weighted squared residuals, often written Δχ². It is not a probability. Extra printed digits identify the calculation; they are not a period uncertainty. A publication still needs an uncertainty and harmonic interpretation appropriate to the data.

Each ZTF validation test also flips residual signs by whole observing night, 100,000 times, taking the maximum over all 20 saved frequencies. This assumes symmetric residuals and independent nights. It does not model every possible instrumental error. Zero exceedances would mean a resolution-limited simulation result, never zero false-alarm probability.

## Check the first star in independent observations

**TESS:** sector 72 uses its 120-second SPOC light curve and target-pixel file. Sectors 22, 45, 46 and 49 use separate 9×9-pixel cutouts, a fixed 1.5-pixel aperture, a background plane fitted to faint outer pixels, and daily offsets/slopes. Other aperture sizes and trend lengths are sensitivity checks. Only quality-zero, finite observations are retained. The ZTF frequencies stay fixed; TESS sectors have their own amplitudes and phases.

The four cutout sectors give improvements of 16.54, 23.02, 41.90 and 30.81. Their strongest varying pixels lie within 0.71 pixel of the target. Sector 72 gives a PDC improvement of 21.39. Sector 72's light curve, pixel extractions, SAP and PDC products share photons and are not counted as separate observations.

Whole-block noise tests use 0.2-, 0.5- and 1-day blocks, 100,000 draws each. All block lengths are reported. Pixel covariance, background subtraction, correlated noise and block independence remain limitations. Cutout timing uses a target-directed geocentric correction and omits TESS's orbital light time, at most approximately 1.5 seconds in this approximation.

**SDSS spectra:** 25 visit files contain 96 exposures over 24 nights. The first 16 nights select a feature and fit its amplitude/phase; the final eight test it. Each exposure lasts 900 seconds. The original broad g−r spectral-colour test has too few usable measurements after masks; that failure is preserved.

A later test compares 31 fixed local spectral windows at the 20 saved frequencies. Early spectra select the 4,060–4,140 Å window, with sidebands at 4,000–4,040 and 4,160–4,200 Å. It has 58 accepted early exposures and 34 later exposures. The later prediction improves the fit by 18.38 without changing the early amplitude or phase. The number “4100” is a wavelength label, not an identification of an atomic line.

Eight comparison stars observed in the same exposures check common calibration changes. Subtracting or fitting their shared variation retains the target signal. Stricter masks, extra error, a curved continuum and within-visit slopes are also tested. These reuse the same observations. The fixed-feature noise simulations do not repeat the 620-way training selection and are not discovery false-alarm probabilities.

## Check which source varies around the second star

The fixed 3.431-day period is tested in g-band ZTF difference images from field 770. Of 399 planned image sets, 359 have the required difference image, mask and point-spread function. The missing sets stay missing; they are not filled in or counted as failures of the star.

The extraction fits 19 Gaia sources simultaneously, with proper-motion-adjusted positions and a sky plane. A second model adds positional derivatives for nearby sources. The clean-source pixel mask is 6141; bits indicating legitimate detected sources are not incorrectly discarded. Pair covariance is retained for the closest neighbour, about two arcseconds away.

The target's full-data improvements are 86.63 and 60.34 in the two models. Fitting early image measurements predicts the later ones, with improvements 21.97 and 19.57. Artificial-source trials vary the point-spread-function width and centroid to measure leakage between the pair. These only bound the tested perturbations.

Image fits reuse ZTF photons. They support source attribution but do not provide independent photometric confirmation. The available crowded TESS measurement was not accepted as such a confirmation.

## What is left open

We have not established rotation rather than another source of variability, a binary companion, a dynamical mass, surface geometry, or unusual chemistry. Earlier toy atmosphere fits were inadequate and supplied no such discovery. Those physical-model experiments are outside this reproduction's claims and scope.

The result is exploratory and the noise assumptions are imperfect. Catalogue searches support apparent novelty but cannot prove that no earlier report exists. An independent specialist should review the calculations, aliases, timing, selection history and prior literature before a scientific publication makes a priority claim.

The archived scripts retain their original names, including names of other stars whose generic routines they supply. [The file guide](FILES.md) identifies their roles. The runner supplies the correct target. It never loads archived fit results into the calculation; archived results are read only after the run to check agreement.
