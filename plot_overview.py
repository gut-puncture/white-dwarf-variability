"""Show the early/later ZTF comparison using freshly prepared measurements."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    output = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(output / "scripts"))
    from fast_rotation_search import SplitModel

    panels = [(80364427, "zg", "117.4-minute star · green light"),
              (80364427, "zr", "117.4-minute star · red light"),
              (65258778, "zg", "3.431-day candidate · green light")]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), layout="constrained")
    for ax, (sid, band, title) in zip(axes, panels):
        d = pd.read_csv(output / f"data/fast_rotation/{sid}_prepared.csv.gz", dtype={"oid": str})
        chosen = json.loads((output / f"data/fast_rotation/{sid}_frozen_discovery_candidates.json").read_text())
        frequency, tref = chosen["selected"][0]["frequency"], chosen["tref"]
        for split, label, colour in [("discovery", "Earlier nights", "#285b8a"), ("validation", "Later nights", "#c6611b")]:
            part = d[(d.filtercode == band) & (d.split == split)]
            model = SplitModel(part, tref)
            phase = (frequency * model.t) % 1
            for k in range(12):
                mask = np.floor(phase * 12) == k
                if not mask.any():
                    continue
                weight = 1 / model.error[mask] ** 2
                magnitude = np.average(model.residual[mask], weights=weight)
                flux_ratio = 10 ** (-0.4 * magnitude)
                error = 100 * 0.4 * np.log(10) * flux_ratio / np.sqrt(weight.sum())
                ax.errorbar((k + 0.5) / 12, 100 * (flux_ratio - 1), error, fmt="o", markersize=4,
                            color=colour, label=label if k == 0 else None)
            # The line is fitted only to earlier nights, not retuned to later dots.
            if split == "discovery":
                fit = model.fit(frequency, True)
                x = np.linspace(0, 1, 250)
                a, b = fit["coefficients"]
                curve = a * np.sin(2 * np.pi * x) + b * np.cos(2 * np.pi * x)
                ax.plot(x, 100 * (10 ** (-0.4 * curve) - 1), color=colour, linewidth=1.5)
        ax.axhline(0, color="#d2d2d2", linewidth=0.7, zorder=0)
        ax.set(title=title, xlabel="Position within one repeating cycle", ylabel="Brightness change (%)", xlim=(-0.03, 1.03))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("A cycle found in earlier nights also appears in later nights", fontsize=16)
    fig.savefig(output / "figures/overview.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
