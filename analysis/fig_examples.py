"""One example fall segment per dataset: motion map + static amplitude map of the most active radar."""
import numpy as np
from common import *

ev, co = events(), composition()
pick = []
for d in ["Dataset 1", "Dataset 2", "Dataset 3"]:
    r = ev[(ev.dataset == d) & (ev.quality == "normal")].iloc[3]
    pick.append(r)
fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), gridspec_kw={"width_ratios": [1, 1, 0.9]})
for i, r in enumerate(pick):
    F = feats(r.stream_id)
    seg = co[co.segment_id == r.event_id].iloc[0]
    t0, t1 = seg.t_start - 4, seg.t_end + 4
    mm = F["mot_map"].astype(np.float32)
    rad = int(np.argmax(mm[:, int(r.t_start * 5):int(r.t_end * 5)].mean((1, 2))))
    m = mm[rad, int(t0 * 5):int(t1 * 5)].T
    ax = axes[i, 0]
    ax.imshow(np.log10(m + 1e-6), aspect="auto", origin="lower", cmap=SEQ, extent=[t0, t1, 0, 184],
              vmin=np.percentile(np.log10(m + 1e-6), 5), vmax=np.percentile(np.log10(m + 1e-6), 99.5))
    a = F["amp_1s"][rad, int(t0):int(t1)].T
    ax2 = axes[i, 1]
    ax2.imshow(a, aspect="auto", origin="lower", cmap=SEQ, extent=[int(t0), int(t1), 0, 184],
               vmax=np.percentile(a, 99), interpolation="nearest")
    for ax_ in (ax, ax2):
        ax_.grid(False)
        ax_.axvspan(r.t_start, r.t_end, color=C[1], alpha=0.0)
        for x in (r.t_start, r.t_end):
            ax_.axvline(x, color=C[1], lw=1.4)
        for x in (seg.t_start, seg.t_end):
            ax_.axvline(x, color=INK, lw=1, ls=":")
        ax_.set_ylabel("range bin")
    ax.set_title(f"{r.dataset} · {r.fall_type} · {r.stream_id} · {RADARS[rad]}\nmotion |residual| (log)", fontsize=9)
    ax2.set_title("static amplitude |mean CIR| per second", fontsize=9)
    e = F["e_mot"][rad].astype(np.float64)
    tt = np.arange(int(t0 * 50), int(t1 * 50)) / 50
    ax3 = axes[i, 2]
    ax3.plot(tt, np.convolve(e[int(t0 * 50):int(t1 * 50)], np.ones(10) / 10, "same"), color=C[0], lw=1.2)
    ax3.set_yscale("log")
    ax3.axvspan(r.t_start, r.t_end, color=C[1], alpha=0.25, lw=0, label="annotated fall")
    for x in (seg.t_start, seg.t_end):
        ax3.axvline(x, color=INK, lw=1, ls=":")
    ax3.set_title("motion energy (strongest radar)", fontsize=9)
    ax3.legend(loc="upper right")
for ax in axes[-1]:
    ax.set_xlabel("stream time (s)")
fig.suptitle("Orange = annotated fall bounds · dotted = segment junctions (fall recording spliced into the stream)", fontweight="bold")
fig.tight_layout()
save(fig, "06_fall_examples")
