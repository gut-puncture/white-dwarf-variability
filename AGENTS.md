# Working on this repository

Read README.md, then docs/METHOD.md. Run `python3 reproduce.py` from the repository root. It creates its own environment and outputs; no credentials or paid service are needed.

Keep original observations, observing plans and archived reference results unchanged. Do not change expected numbers or tolerances just to pass a failed comparison. Investigate and document any disagreement. Report incomplete runs as incomplete.

The run covers two targets, not all 400 screened stars. The 117.4-minute result has independent TESS support and spectral variation. The 3.431-day result remains a candidate; ZTF image extraction reuses ZTF photons. The first star already had an ATLAS dubious-variable flag and known magnetism. Do not claim global novelty, a physical cause, or independent expert review.

`reference/` is for comparison after computation. Do not feed reference fit results into the reproduction. Preserve failed tests and missing observations. New scientific analyses must be labelled separately from reproduction of the recorded work.

Run `python3 -m unittest discover -s tests` in the installed environment for failure-handling tests. The full numerical verification is `reproduce.py`; unit tests alone do not validate the astronomy.
