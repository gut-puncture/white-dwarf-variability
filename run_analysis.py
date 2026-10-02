"""Run the original measurements in dependency order, without saved fit outputs."""
from pathlib import Path
import os
import subprocess
import sys
import time


def main():
    root = Path(sys.argv[1]).resolve()
    scripts = root / "scripts"
    stages = [
        ("Search ZTF for both periods", "code", "import fast_rotation_search as s; import pandas as pd; p=pd.read_csv(s.ROOT/'data/fast_rotation_extension_target_plan.csv',dtype={'gaia_dr3_source_id':str}).fillna({'known_control':''}); [s.one(p.set_index('sdss_id',drop=False).loc[sid].to_dict(),1440.,400) for sid in [80364427,65258778]]"),
        ("Check the first star against later ZTF nights", "fast_rotation_candidate_ztf_audit.py", "--sid", "80364427", "--null-draws", "100000", "--family-size", "400"),
        ("Check the second star against later ZTF nights", "fast_rotation_candidate_ztf_audit.py", "--sid", "65258778", "--null-draws", "100000", "--family-size", "400"),
        ("Measure TESS sector 72", "fast_rotation_tess_validation.py", "--sid", "80364427"),
        ("Locate the signal in sector 72 pixels", "code", "import importlib; m=importlib.import_module('109631850_pixel_and_spectrum_checks'); m.SID=80364427; m.pixels()"),
        ("Test sector 72 against correlated noise", "80364427_tess_correlated_null.py"),
        ("Extract and fit four more TESS sectors", "80364427_additional_tess_validation.py"),
        ("Locate and test the four TESS signals", "80364427_additional_tess_audit.py"),
        ("Repeat the unsuccessful broad-colour test", "80364427_spectral_colours.py"),
        ("Measure fixed spectral windows", "80364427_continuum_windows.py"),
        ("Choose the spectral feature on early nights and test later nights", "80364427_spectral_feature_scan.py"),
        ("Check comparison stars from the same exposures", "80364427_feature_controls.py"),
        ("Change spectral errors, masks and continuum fits", "80364427_feature_sensitivity.py"),
        ("Measure the second star and its neighbours in ZTF images", "65258778_ztf_image_photometry.py"),
        ("Test image attribution and artificial-source leakage", "65258778_ztf_image_period.py"),
    ]
    for index, (label, program, *arguments) in enumerate(stages, 1):
        print(f"[{index}/{len(stages)}] {label}", flush=True)
        command = [sys.executable, "-c", arguments[0]] if program == "code" else [sys.executable, str(scripts / program), *arguments]
        log = root / "logs" / f"{index:02d}.log"
        started = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=scripts, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"  Still running ({time.monotonic() - started:.0f}s); details in logs/{index:02d}.log", flush=True)
        if process.returncode:
            print(log.read_text()[-5000:], file=sys.stderr)
            raise SystemExit(process.returncode)
        print(f"  Done ({time.monotonic() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
