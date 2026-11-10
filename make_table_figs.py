import re, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import style; style.apply(); P = style.PAL
OUT = ["$C_{f,1}$", "$C_{f,2}$", r"$\mathrm{Nu}_1$", r"$\mathrm{Nu}_2$"]
MOD = ["MLP ens.", "HistGB", "ANFIS-GA", "ANFIS-PSO"]; MC = [P["mlp"], P["gb"], P["ga"], P["pso"]]
# ---- five-split mean/SD/min (Table multiseed)
MS = {"mlp": [(.9802,.0035,.9754),(.9694,.0039,.9636),(.9975,.0007,.9966),(.9970,.0005,.9963)],
      "gb":  [(.9073,.0116,.8933),(.9172,.0099,.9093),(.9790,.0014,.9774),(.9807,.0015,.9786)],
      "ga":  [(.8914,.0135,.8723),(.8642,.0265,.8394),(.9861,.0091,.9703),(.9884,.0016,.9861)],
      "pso": [(.8615,.0142,.8459),(.8258,.0275,.7787),(.6407,.7767,-.7487),(.9776,.0206,.9409)]}
keys = ["mlp","gb","ga","pso"]
# F1: heat-map of mean R2
M = np.array([[MS[k][j][0] for j in range(4)] for k in keys])
cmap = LinearSegmentedColormap.from_list("t", ["#FBE9E7", "#F6C28B", "#9ED8CF", "#0B7A75"])
fig, ax = plt.subplots(figsize=(5.6, 2.9)); ax.grid(False)
im = ax.imshow(np.clip(M, .6, 1), cmap=cmap, vmin=.6, vmax=1, aspect="auto")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{M[i,j]:.3f}\n±{MS[keys[i]][j][1]:.3f}", ha="center", va="center", fontsize=8, color=P["ink"])
ax.set_xticks(range(4)); ax.set_xticklabels(OUT); ax.set_yticks(range(4)); ax.set_yticklabels(MOD)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0); fig.colorbar(im, ax=ax, label=r"mean test $R^2$ (clipped at 0.6)", fraction=.04)
fig.savefig("figures/fig_heat_models.png"); plt.close(fig)
# F2: worst-case (min) R2 per split: dumbbell mean vs min
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.7), sharey=True)
for j, ax in enumerate(axs):
    for i, k in enumerate(keys):
        m, s, mn = MS[k][j]
        ax.plot([max(mn, -0.1), m], [i, i], color=MC[i], lw=2, alpha=.5)
        ax.scatter([m], [i], s=45, color=MC[i], zorder=3)
        ax.scatter([max(mn, -0.1)], [i], s=45, marker="v" if mn >= -0.1 else "X", facecolor="white", edgecolor=MC[i], lw=1.5, zorder=3)
        if mn < -0.1: ax.annotate(f"min {mn:.2f}", (-0.1, i), xytext=(4, 6), textcoords="offset points", fontsize=7, color=MC[i])
    ax.set_title(OUT[j]); ax.set_yticks(range(4)); ax.set_yticklabels(MOD if j == 0 else [])
    ax.set_xlim(-0.15, 1.02); ax.invert_yaxis(); ax.set_xlabel(r"$R^2$")
fig.suptitle("Mean (filled) and worst (open) test $R^2$ over five splits", y=1.04, fontsize=9, color=P["ink"]); fig.tight_layout()
fig.savefig("figures/fig_worstcase.png"); plt.close(fig)
# F3: learning curve, symlog
n = [18,50,150,500,1500,2546]
LC = {"Cf1": {"pso":[.165,-.474,.296,-703.904,.818,.782], "ga":[.251,-.328,.295,.840,.849,.843], "mlp":[.311,.307,.432,.746,.950,.984]},
      "Nu1": {"pso":[.212,.042,-510.494,.982,.987,.987], "ga":[.081,.539,.940,.981,.990,.989], "mlp":[.398,.842,.911,.975,.996,.999]}}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, (nm, tt) in zip(axs, [("Cf1", OUT[0]), ("Nu1", OUT[2])]):
    for k, lab, c, mk in [("mlp","MLP ens.",P["mlp"],"o"),("ga","ANFIS-GA",P["ga"],"s"),("pso","ANFIS-PSO",P["pso"],"^")]:
        ax.plot(n, LC[nm][k], marker=mk, color=c, lw=1.6, ms=5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1, linscale=1.2); ax.axhline(0, color=P["mute"], lw=.8)
    ax.set_yticks([-1000,-100,-10,-1,0,1]); ax.set_yticklabels(["$-10^3$","$-10^2$","$-10$","$-1$","0","1"])
    ax.set_title(tt); ax.set_xlabel("training samples $n$")
axs[0].set_ylabel(r"test $R^2$ (symlog axis)"); axs[0].legend(loc="lower right", fontsize=7.5)
fig.tight_layout(); fig.savefig("figures/fig_learning_symlog.png"); plt.close(fig)
# F4: lollipop importance
IMP = {OUT[0]: [("$\\mathcal{A}_1$",1.08),("$\\Lambda_s$",.79),("$\\beta_c$",.68),("$\\mathcal{A}_2$",.68),("$\\Omega_r$",.31)],
       OUT[1]: [("$\\mathcal{A}_2$",1.54),("$\\Lambda_s$",.76),("$\\beta_c$",.68),("$\\mathcal{A}_1$",.68),("$\\Omega_r$",.42)],
       OUT[2]: [("Rd",1.30),("$\\mathcal{A}_1$",.89),("$\\mathcal{A}_2$",.77),("Re",.46),("$\\Lambda_s$",.14)],
       OUT[3]: [("Rd",1.27),("$\\mathcal{A}_2$",1.00),("$\\mathcal{A}_1$",.59),("Re",.41),("$\\Lambda_s$",.18)]}
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.4), sharex=True)
for ax, (o, L) in zip(axs, IMP.items()):
    y = np.arange(5)[::-1]; v = [x[1] for x in L]
    ax.hlines(y, 0, v, color=P["mute"], lw=1.5); ax.scatter(v, y, s=60, color=P["mlp"], zorder=3, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels([x[0] for x in L]); ax.set_title(o); ax.set_xlabel("RMSE increase"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("figures/fig_importance_lollipop.png"); plt.close(fig)
# F5: conformal coverage
cov = [94.2,94.9,95.3,96.9]; q = [2.517,2.806,3.062,3.129]; se = 100*np.sqrt(.95*.05/450)
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.axhspan(95-se, 95+se, color=P["mlp"], alpha=.15, lw=0); ax.axhline(95, color=P["mlp"], lw=1, ls="--")
ax.scatter(range(4), cov, s=70, color=P["ga"], zorder=3, edgecolor="white")
for i in range(4): ax.annotate(f"$q$={q[i]:.2f}", (i, cov[i]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5)
ax.set_xticks(range(4)); ax.set_xticklabels(OUT); ax.set_ylim(93, 98); ax.set_ylabel("empirical coverage (%)")
ax.text(3.45, 95+se+.1, "nominal 95% $\\pm1$ SE", ha="right", fontsize=7.5, color=P["mlp"])
fig.tight_layout(); fig.savefig("figures/fig_coverage.png"); plt.close(fig)
# F6: base-study settings (mean R2 +- SD) dumbbell
BS = {OUT[0]: dict(pm=(.8615,.0142),pb=(-.5791,3.2655),gm=(.8914,.0135),gb=(.9089,.0108)),
      OUT[1]: dict(pm=(.8258,.0275),pb=(.6535,.2998),gm=(.8642,.0265),gb=(.8744,.0199)),
      OUT[2]: dict(pm=(.6407,.7767),pb=(.9842,.0091),gm=(.9861,.0091),gb=(.9914,.0022)),
      OUT[3]: dict(pm=(.9776,.0206),pb=(.9878,.0024),gm=(.9884,.0016),gb=(.9906,.0012))}
fig, ax = plt.subplots(figsize=(6.2, 3.0)); xs = np.arange(4)
for off, key, c, mk, lab in [(-.27,"pm",P["pso"],"o","PSO, main"),(-.09,"pb",P["pso"],"D","PSO, base-study"),(.09,"gm",P["ga"],"o","GA, main"),(.27,"gb",P["ga"],"D","GA, base-study")]:
    v = np.array([BS[o][key][0] for o in OUT]); ax.scatter(xs+off, np.clip(v, .55, 1), s=40, color=c, marker=mk, facecolor=c if mk=="o" else "white", edgecolor=c, lw=1.4, label=lab, zorder=3)
    for i, o in enumerate(OUT):
        if BS[o][key][0] < .55: ax.annotate(f"{BS[o][key][0]:.2f}\n(off-scale)", (xs[i]+off, .55), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.5, color=c)
ax.set_xticks(xs); ax.set_xticklabels(OUT); ax.set_ylim(.55, 1.01); ax.set_ylabel("mean test $R^2$ (5 splits)")
ax.legend(ncol=2, fontsize=7.5, loc="lower right"); fig.tight_layout(); fig.savefig("figures/fig_optimiser_settings.png"); plt.close(fig)
# F7: surrogate error on verified designs
fig, ax = plt.subplots(figsize=(3.8, 2.4)); lab = [r"$\mathrm{Nu}_1$", r"$C_{f,1}$", r"$\Gamma_{\min}$"]; val = [3.76, 17.9, 5.88]
ax.barh(lab[::-1], val[::-1], color=[P["gb"], P["pso"], P["mlp"]], height=.55)
for i, v in enumerate(val[::-1]): ax.text(v+.4, i, f"{v:.2f}%", va="center", fontsize=8)
ax.set_xlabel("mean relative surrogate error (%)"); ax.set_xlim(0, 22); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("figures/fig_design_error.png"); plt.close(fig)
# F8: base study vs present (parsed from old Table basecomp)
tx = open("main.tex", encoding="utf8").read(); i0 = tx.index("\\label{tab:basecomp}"); blk = tx[i0:tx.index("\\end{table*}", i0)]
rows = re.findall(r"([0-9.]+) / ([0-9.]+)", blk); a = np.array(rows, float); print("pairs", a.shape)
assert a.shape == (96, 2)
a = a.reshape(24, 4, 2); base = a[:, :, 0]; pres = a[:, :, 1]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
for ax, cols, ttl, c in [(axs[0], [0,2], "skin friction (lower & upper disk)", P["mlp"]), (axs[1], [1,3], "local Nusselt number (lower & upper disk)", P["pso"])]:
    b, p = base[:, cols].ravel(), pres[:, cols].ravel(); lo, hi = min(b.min(), p.min())*.8, max(b.max(), p.max())*1.2
    ax.plot([lo, hi], [lo, hi], color=P["ink"], lw=1, ls=":"); ax.scatter(b, p, s=26, color=c, alpha=.8, edgecolor="white", lw=.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_title(ttl, fontsize=8.5)
    import matplotlib.ticker as mt
    tk = [t for t in (2,3,5,10,20,30,50) if lo <= t <= hi]
    for axis in (ax.xaxis, ax.yaxis): axis.set_major_locator(mt.FixedLocator(tk)); axis.set_major_formatter(mt.FuncFormatter(lambda v,_: f"{v:g}")); axis.set_minor_locator(mt.NullLocator())
    ax.set_xlabel("Upreti et al. (Table 2)"); ax.set_ylabel("present solver, reduced model")
    print(ttl, "median |rel diff| %:", np.median(abs(p-b)/b*100).round(2), "range", (abs(p-b)/b*100).min().round(2), (abs(p-b)/b*100).max().round(2))
fig.tight_layout(); fig.savefig("figures/fig_base_comparison.png"); plt.close(fig)
print("done")
