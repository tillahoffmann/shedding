from glob import glob
import json
from matplotlib import pyplot as plt
import numpy as np


def to_float(x):
    return np.nan if x is None else x


np.random.seed(0)
spread = 0.45
arrow_kwargs = {
    "head_length": 0.1,
    "head_width": 0.1,
}
lim_length = 0.25
filenames = glob("publications/*/*.json")

fig, ax = plt.subplots()
ylabels = []
for i, filename in enumerate(sorted(filenames, reverse=True)):
    color = f"C{i}"

    with open(filename) as fp:
        publication = json.load(fp)
    ylabels.append(publication["key"])

    # Plot the level of quantification
    loq = publication["loq"]
    if loq is not None:
        ax.plot((loq, loq), (i - spread, i + spread), ls=":", color="k")

    # Plot individual measurements
    loads = publication.get("loads")
    if loads:
        x = np.random.permutation([to_float(load["value"]) for load in loads])
        y = i + np.linspace(-spread, spread, len(x))
        ax.scatter(x, y, color=color, marker=".", alpha=0.5)

    # Plot limits
    summaries = publication.get("load_summaries", {})
    extremum_markers = []
    if "max" in summaries:
        ax.arrow(summaries["max"], i, -lim_length, 0, color=color, **arrow_kwargs)
        extremum_markers.append(summaries["max"])
    if "min" in summaries:
        ax.arrow(summaries["min"], i, lim_length, 0, color=color, **arrow_kwargs)
        extremum_markers.append(summaries["min"])
    if extremum_markers:
        ax.scatter(
            extremum_markers,
            i * np.ones_like(extremum_markers),
            color=color,
            marker="|",
        )

    # Plot other summary statistics
    for key, marker in [("mean", "s"), ("median", "^")]:
        if key in summaries:
            ax.scatter(summaries[key], i, marker=marker, color=color)

ax.yaxis.set_ticks(np.arange(len(ylabels)))
ax.yaxis.set_ticklabels(ylabels)
ax.set_xlabel(r"$\log_{10}$ copies per mL")
fig.tight_layout()
