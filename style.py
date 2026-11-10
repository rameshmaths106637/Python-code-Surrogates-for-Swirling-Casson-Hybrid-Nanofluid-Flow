import matplotlib as mpl
PAL = dict(mlp="#0B7A75", gb="#E09F3E", ga="#6A4C93", pso="#E4572E", ink="#22223B", grid="#D9D9E3", mute="#8D99AE")
def apply():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
        "axes.edgecolor": PAL["ink"], "axes.labelcolor": PAL["ink"], "xtick.color": PAL["ink"], "ytick.color": PAL["ink"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": PAL["grid"],
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 130, "savefig.dpi": 220,
        "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 9.5})
