import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.cm as cm
from matplotlib.colors import Normalize
from independent_solver import solve
import style; style.apply()
xi = np.linspace(0, 1, 201)
# (key, label, values, colormap, transform)
SW = {
 "Re":  (r"$\mathrm{Re}$", [0.5,1,2,3,5], "magma_r"),
 "Om":  (r"$\Omega_r$", [-1,-0.5,0,0.5,1], "cividis"),
 "A1":  (r"$\mathcal{A}_1$", [-1,-0.5,0,0.5,1], "viridis"),
 "A2":  (r"$\mathcal{A}_2$", [-1,-0.5,0,0.5,1], "mako" if False else "GnBu"),
 "M":   (r"$\mathrm{M}$", [0,1,2,3,4], "plasma"),
 "chi": (r"$\chi_e$", [0,0.5,1,1.5], "PuRd"),
 "beta":(r"$\beta_c$", [0.5,1,1.5,3,5], "YlOrBr"),
 "Lam": (r"$\Lambda_s$", [0,0.1,0.2,0.3], "BuPu"),
 "Rd":  (r"$\mathrm{Rd}$", [0,0.5,1,2,3], "inferno"),
 "gt":  (r"$\gamma_t$", [0,0.01,0.02,0.035,0.05], "YlGn"),
 "Ec":  (r"$\mathrm{Ec}$", [0,0.01,0.02,0.035,0.05], "copper_r"),
 "phi": (r"$\phi_1+\phi_2$", [0,0.02,0.04,0.06,0.08], "Blues"),
}
def prof(key, v):
    kw = {"phi1": v/2, "phi2": v/2} if key == "phi" else {key: v}
    s = solve(**kw)["sol"]; y = s.sol(xi); return y[1], y[4], y[6]
FIGS = [("fig_prof_A.png", ["Re","Om"]), ("fig_prof_B.png", ["A1","A2"]), ("fig_prof_C.png", ["M","chi"]),
        ("fig_prof_D.png", ["beta","Lam"]), ("fig_prof_E.png", ["Rd","gt"]), ("fig_prof_F.png", ["Ec","phi"])]
ylabs = [r"$\mathcal{F}'(\xi)$", r"$\mathcal{G}(\xi)$", r"$\Theta(\xi)$"]
for fn, keys in FIGS:
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 7.4), sharex=True)
    for c, key in enumerate(keys):
        lab, vals, cmap = SW[key]; cmo = plt.get_cmap(cmap); norm = Normalize(min(vals), max(vals))
        for v in vals:
            col = cmo(0.25 + 0.7*norm(v)) if max(vals) > min(vals) else cmo(0.6)
            try: P = prof(key, v)
            except Exception as e: print("fail", key, v, e); continue
            for r in range(3):
                axs[r, c].plot(xi, P[r], color=col, lw=2.0, solid_capstyle="round")
        for r in range(3):
            axs[r, c].set_ylabel(ylabs[r] if c == 0 else "")
            axs[r, c].margins(x=0)
        axs[0, c].set_title(lab + "  (" + ", ".join(f"{x:g}" for x in vals) + ")")
        axs[2, c].set_xlabel(r"$\xi = z/d$")
        sm = cm.ScalarMappable(norm=Normalize(0.25, 0.95), cmap=cmo)
        cb = fig.colorbar(sm, ax=axs[:, c], orientation="horizontal", fraction=0.03, pad=0.08, aspect=40)
        cb.set_ticks([0.25, 0.95]); cb.set_ticklabels([f"{min(vals):g}", f"{max(vals):g}"]); cb.ax.tick_params(length=0, labelsize=8)
    fig.savefig("figures/" + fn); plt.close(fig); print("saved", fn, flush=True)
