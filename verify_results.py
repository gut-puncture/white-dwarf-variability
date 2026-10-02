"""Compare newly calculated quantities with the archived research results."""
from pathlib import Path
import json
import math
import sys

ROOT = Path(__file__).resolve().parent


def measurements(data):
    def read(name):
        return json.loads((data / f"{name}.json").read_text())

    values = {}
    for sid in (80364427, 65258778):
        result = read(f"fast_rotation/{sid}_result")
        best = result["checks"][0]
        for key in ("n_raw", "n_clean", "n_discovery", "n_validation"):
            values[f"{sid}.ztf.{key}"] = result[key]
        values[f"{sid}.period_seconds"] = best["period_seconds"]
        values[f"{sid}.ztf.training_gain"] = best["discovery"]["delta_chi2"]
        values[f"{sid}.ztf.validation_gain"] = best["validation"]["delta_chi2"]
        values[f"{sid}.ztf.prediction_gain"] = best["validation_gain_with_discovery_amplitude_phase"]
        audit = read(f"{sid}_fast_rotation_ztf_audit")["validation_night_null"]
        values[f"{sid}.ztf.null_draws"] = audit["draws"]
        values[f"{sid}.ztf.null_exceedances"] = audit["exceedances"]

    tess = read("80364427_fast_rotation_tess_validation")
    primary = next(r for r in tess["tests"] if r["sector"] == 72 and r["discovery_rank"] == 1)
    values["80364427.tess.72.gain"] = primary["pdc"]["delta_chi2"]
    pixels = read("80364427_tess_pixel_checks")["sectors"][0]
    values["80364427.tess.72.pixel_offset"] = pixels["maximum_pixel"]["distance_pixels"]
    for r in read("80364427_additional_tess_validation")["tests"]:
        if r["frequency_family"] == "original" and r["discovery_rank"] == 1 and r["aperture_radius"] == 1.5 and r["nuisance_segment_days"] == 1:
            values[f"80364427.tess.{r['sector']}.gain"] = r["delta_chi2"]
            values[f"80364427.tess.{r['sector']}.n"] = r["n"]
    audit = read("80364427_additional_tess_audit")
    for r in audit["pixel_checks"]:
        values[f"80364427.tess.{r['sector']}.pixel_offset"] = r["maximum"]["distance_to_target_pixels"]
    for r in audit["block_nulls"]:
        prefix = f"80364427.tess.null_{r['block_days']}day"
        values[prefix + ".gain"] = r["combined"]["observed"]
        values[prefix + ".draws"] = r["draws"]
        values[prefix + ".exceedances"] = r["combined"]["exceedances"]

    feature = read("80364427_spectral_features_validation")["selected_validation"]
    values["80364427.spectra.selected_window"] = feature["feature"]
    for split in ("training", "validation", "validation_unchanged_amplitude_phase"):
        for key in ("n", "n_nights", "delta_chi2"):
            values[f"80364427.spectra.{split}.{key}"] = feature[split][key]
    for r in read("80364427_feature_controls")["tests"]:
        if r["feature"] == "4100":
            values[f"80364427.spectra.control_{r['mode']}.prediction_gain"] = r["validation_unchanged_amplitude_phase"]["delta_chi2"]
    for r in read("80364427_feature_sensitivity")["tests"]:
        values[f"80364427.spectra.sensitivity_{r['variant']}.prediction_gain"] = r["validation_unchanged_prediction"]["delta_chi2"]
    values["80364427.spectra.broad_colour_test"] = read("80364427_spectral_colour_tests")["status"]

    images = read("65258778_ztf_image_period")
    values["65258778.images.planned"] = images["n_planned_images"]
    values["65258778.images.extracted"] = images["n_extracted_images"]
    for r in images["scalar_results"]:
        if r["source_id"] not in ("1974721783974773376", "1974720310808109696"):
            continue
        name = "target" if r["source_id"] == "1974721783974773376" else "neighbour"
        prefix = f"65258778.images.{name}.{r['mode']}"
        if r["subset"] == "full":
            values[prefix + ".gain"] = r["delta"]
            values[prefix + ".n"] = r["n"]
        if r["subset"] == "chronological":
            values[prefix + ".prediction_gain"] = r["unchanged_prediction_gain"]
    return values


def compare(actual, expected):
    checks = []
    for key, wanted in expected.items():
        value = actual.get(key)
        # Rounding tolerance for platform-dependent linear algebra, not a
        # scientific confidence interval. Counts and labels must match exactly.
        tolerance = 0 if isinstance(wanted, (str, int)) else 0.05
        if key.endswith("period_seconds"):
            tolerance = 0.02 if key.startswith("80364427") else 0.5
        if key.endswith("pixel_offset"):
            tolerance = 0.005
        if isinstance(wanted, str):
            passed = value == wanted
        else:
            passed = isinstance(value, (float, int)) and math.isfinite(value) and abs(value - wanted) <= tolerance
        checks.append(dict(measurement=key, expected=wanted, actual=value, absolute_tolerance=tolerance, passed=passed))
    return checks


def main():
    output = Path(sys.argv[1]).resolve()
    actual = measurements(output / "data")
    expected = measurements(ROOT / "reference/data")
    checks = compare(actual, expected)
    report = {"passed": all(r["passed"] for r in checks), "checks": checks}
    (output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "measurements.json").write_text(json.dumps(actual, indent=2, allow_nan=False) + "\n")
    failures = [r for r in checks if not r["passed"]]
    if failures:
        print(json.dumps(failures, indent=2), file=sys.stderr)
        raise SystemExit("The fresh calculation differs from the archived results. See verification.json.")
    a, b = "80364427", "65258778"
    summary = f"""# Reproduction passed

The observations reproduce a **{actual[a+'.period_seconds']/60:.3f}-minute brightness cycle** in SDSS J112148.80+103934.2 and a **{actual[b+'.period_seconds']/86400:.5f}-day candidate cycle** in WDJ215009.13+471257.26.

The first cycle also appears in five TESS observing sectors and in SDSS spectral changes. The second still needs independent photometric confirmation.

| Check | First star | Second star |
| --- | ---: | ---: |
| Accepted ZTF measurements | {actual[a+'.ztf.n_clean']} | {actual[b+'.ztf.n_clean']} |
| Fit improvement on later ZTF nights | {actual[a+'.ztf.validation_gain']:.2f} | {actual[b+'.ztf.validation_gain']:.2f} |
| Improvement with the earlier amplitude and phase unchanged | {actual[a+'.ztf.prediction_gain']:.2f} | {actual[b+'.ztf.prediction_gain']:.2f} |

These numbers are reductions in weighted squared residuals. They are not probabilities or sigma values.

The selected spectral feature improves the later-night fit by **{actual[a+'.spectra.validation_unchanged_amplitude_phase.delta_chi2']:.2f}** without changing its earlier amplitude or phase. The second star has **{actual[b+'.images.extracted']} usable image sets out of {actual[b+'.images.planned']} planned**. Its image fits reuse the catalogue's photons.

The broad spectral-colour test failed because too few measurements survived the quality cuts. The default run repeats that failure.

All {len(checks)} recorded numerical checks passed. See [verification.json](verification.json), [measurements.json](measurements.json), [figures/](figures/) and [logs/](logs/).

This verifies the calculation on the supplied observations. It does not independently validate the method, prove global novelty, or establish the physical cause of either cycle. The first star was already a known magnetic white dwarf and an ATLAS dubious-variable candidate.
"""
    (output / "SUMMARY.md").write_text(summary)
    print(f"PASS: {len(checks)} numerical checks; regenerated from original observations.", flush=True)


if __name__ == "__main__":
    main()
