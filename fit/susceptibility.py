# Per-gene susceptibility s_g of the M1 model from the pooled viability screen at three TIL:tumor ratios.
# M1: dT_g/dt = (r - k(1 + s_g) E) T_g. Comparing each knockout g to the control guides, and the
# co-culture arm to the IFNγ-only arm (the reference of the bulk killing data), growth and IFNγ cancel:
#   ln e_g(R) = -s_g * kappa * R * t,   e_g = (n_g / n_ctrl)_coculture / (n_g / n_ctrl)_IFNγ
# s_g < 0: resistant (s_g = -1 is never killed); s_g > 0: sensitized. kappa comes from fit/kill_rate.py.
# Inputs:  Supplementary Tables 2h, 2l, 2p (per-guide MAGeCK, co-culture vs IFNγ, day 17)
# Outputs: results/params/susceptibility.csv, results/figures/susceptibility.png
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import CONTROL_TARGETS, ROOT, load_supp_table

SHEETS = {1: "2h", 2: "2l", 4: "2p"}  # TIL:tumor ratio -> per-guide sheet
HOURS = 48
MIN_READS = 500  # control guides below this are bottlenecked dropouts (LFC sd ~2 vs ~0.35 above)
N_NULL = 2000
rng = np.random.default_rng(0)

kappa = json.load(open(ROOT / "results" / "params" / "kill_rate.json"))["kappa_per_hour_per_ratio"]
ratios = np.array(list(SHEETS))
x = kappa * ratios * HOURS  # expected -ln(survival) of the bulk population at each ratio

# ln e per guide and ratio: MAGeCK log2 fold change, centered on the control-guide median.
lne = {}
for R, sheet in SHEETS.items():
    t = load_supp_table(sheet).set_index("sgrna")
    t = t[t.control_mean >= MIN_READS]
    is_ctrl = t.Gene.isin(CONTROL_TARGETS)
    lne[R] = np.log(2) * (t.LFC - t.LFC[is_ctrl].median())
lne = pd.DataFrame(lne).dropna()  # guides passing the read filter at all three ratios
gene = load_supp_table("2l").set_index("sgrna").Gene.reindex(lne.index)
is_ctrl = gene.isin(CONTROL_TARGETS)


def fit_s(y):
    # y: (..., 3) ln e at each ratio -> s from least squares of y = -s * x through the origin.
    return -(y @ x) / (x @ x)


# Per gene: median ln e over its guides at each ratio, then one s across the three ratios.
per_gene = lne[~is_ctrl].groupby(gene[~is_ctrl]).median()
n_guides = gene[~is_ctrl].value_counts()
res = pd.DataFrame({"s": fit_s(per_gene.to_numpy()), "n_guides": n_guides}, index=per_gene.index)
for R, xi in zip(ratios, x):
    res[f"s_at_{R}to1"] = -per_gene[R] / xi  # single-ratio estimate; flat across ratios if M1 holds

# Null: random triplets of control guides, processed exactly like a gene.
ctrl = lne[is_ctrl].to_numpy()
null = np.array([fit_s(np.median(ctrl[rng.choice(len(ctrl), 3, replace=False)], axis=0)) for _ in range(N_NULL)])
null_single = -np.array([np.median(ctrl[rng.choice(len(ctrl), 3, replace=False)], axis=0) for _ in range(N_NULL)]) / x
res["z"] = res.s / null.std()

# Strongly essential genes (depleted before treatment, Supplementary Table 2a) have few reads per guide
# and give noisy s in both directions, so they are flagged and left out of the hit list.
ess = load_supp_table("2a").set_index("id")
res["essential"] = (ess["neg|fdr"] < 0.05) & (ess["neg|lfc"] < -0.5)
res = res.sort_values("s")

# Checks on the hits (|z| > 3, not essential):
hits = res[(res.z.abs() > 3) & ~res.essential]
s_cols = [f"s_at_{R}to1" for R in ratios]
dose = hits[s_cols].div(hits.s, axis=0).median()  # ~1 at every ratio if ln e scales linearly with R
below_floor = (hits[s_cols] < -1).sum()  # s < -1 means enriched more than a never-killed cell could be
print(f"kappa = {kappa:.4f} /h | guides kept: {len(lne)} ({is_ctrl.sum()} control) | null sd of s: {null.std():.3f}")
print("null sd of single-ratio s:", dict(zip(s_cols, null_single.std(axis=0).round(2))))
print(f"hits with |z| > 3: {len(hits)} (excluded as essential: {((res.z.abs() > 3) & res.essential).sum()})")
print(hits.round(2).to_string())
print("\nsingle-ratio s / fitted s, median over hits (1 = linear in dose):", dose.round(2).to_dict())
print("hits with single-ratio s < -1:", below_floor.to_dict())

(ROOT / "results" / "params").mkdir(parents=True, exist_ok=True)
res.round(4).to_csv(ROOT / "results" / "params" / "susceptibility.csv", index_label="gene")

# Figure: (a) ln e vs expected bulk killing for the hits, with the M1 line for each;
# (b) single-ratio estimates at 2:1 vs 4:1 for all genes, which should fall on the identity line.
fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.4))
xg = np.linspace(0, x.max() * 1.05, 50)
for i, (g, row) in enumerate(hits.iterrows()):
    c = f"C{i % 10}"
    a.plot(x, per_gene.loc[g], "o", color=c, ms=4)
    a.plot(xg, -row.s * xg, color=c, lw=1, label=g)
a.plot(xg, xg, "k--", lw=1, label="never killed (s = -1)")
a.set(xlabel="κ·R·t (bulk −ln survival)", ylabel="ln enrichment vs control guides")
a.legend(frameon=False, fontsize=6, ncol=2)
b.scatter(res[s_cols[1]], res[s_cols[2]], s=6, color="0.6")
b.scatter(hits[s_cols[1]], hits[s_cols[2]], s=10, color="C3")
lim = np.array([-1.6, 1.0])
b.plot(lim, lim, "k--", lw=1)
b.set(xlabel="s from 2:1 only", ylabel="s from 4:1 only", xlim=lim, ylim=lim)
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "susceptibility.png", dpi=200)
