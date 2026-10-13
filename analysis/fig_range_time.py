"""Fall-onset-aligned range-time motion maps (ratio to the ADL median per bin), per dataset and radar."""
import numpy as np
from common import *

z = np.load("/home/user/data_raw/fall_aligned.npz")
W0, W1 = -4, 8
fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), sharex=True, sharey=True)
for i, d in enumerate(["Dataset_1", "Dataset_2", "Dataset_3"]):
    for r in range(3):
        ax = axes[i, r]
        m = np.log10(np.clip(z[d][r].T, 1e-3, None))
        im = ax.imshow(m, aspect="auto", origin="lower", cmap=SEQ, vmin=0, vmax=1.5,
                       extent=[W0, W1, 0, 368], interpolation="nearest")
        ax.axvline(0, color=C[1], lw=1.2, ls="--")
        ax.grid(False)
        if i == 0:
            ax.set_title(RADARS[r])
        if r == 0:
            ax.set_ylabel(f"{d.replace('_', ' ')}\nrange bin")
        if i == 2:
            ax.set_xlabel("time from annotated fall start (s)")
cb = fig.colorbar(im, ax=axes, shrink=0.6, label="log10( motion / ADL median motion )")
fig.suptitle("Average motion around fall onset: each mat position lights up a different radar", fontweight="bold")
save(fig, "05_fall_range_time_maps")
