# Fit the baseline kill rate of the M0 model to bulk killing at three TIL:tumor ratios.
# M0: dT/dt = (r - k*E) T, with E = R*T0 constant. Relative to the no-TIL control, growth r cancels:
#   S(R, t) = T_coculture / T_control = exp(-kappa * R * t),   kappa = k*T0  [per hour per unit ratio]
# Input:  data/paper/killing_48h.csv
# Output: results/params/kill_rate.json, results/figures/kill_rate.png
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import DATA, ROOT

df = pd.read_csv(DATA / "paper" / "killing_48h.csv", comment="#")
R, t = df.ratio_til_to_tumor.to_numpy(float), df.hours.to_numpy(float)
y = -np.log(1 - df.kill_percent.to_numpy() / 100)  # = kappa * R * t under M0


def fit_kappa(R, t, y):
    # Least squares for y = kappa * (R t), line through the origin.
    x = R * t
    return (x @ y) / (x @ x)


kappa = fit_kappa(R, t, y)
per_point = y / (R * t)  # flat across ratios if mass action holds

# Leave one ratio out: fit on two, predict the killing at the third.
loo = []
for i in range(len(df)):
    keep = np.arange(len(df)) != i
    k_i = fit_kappa(R[keep], t[keep], y[keep])
    loo.append(100 * (1 - np.exp(-k_i * R[i] * t[i])))
df["kappa_per_point"] = per_point
df["kill_fit"] = 100 * (1 - np.exp(-kappa * R * t))
df["kill_loo"] = loo
print(df.round(4).to_string(index=False))

resid = y - kappa * R * t
se = np.sqrt((resid @ resid) / (len(y) - 1) / ((R * t) @ (R * t)))
out = {
    "kappa_per_hour_per_ratio": kappa,
    "kappa_se": se,
    "kappa_per_48h_per_ratio": kappa * 48,
    "kappa_per_point": dict(zip(df.ratio_til_to_tumor.astype(str), per_point.round(6))),
    "loo_abs_error_kill_percent": float(np.mean(np.abs(df.kill_loo - df.kill_percent))),
    "data": "data/paper/killing_48h.csv",
    "model": "S = exp(-kappa * R * t), growth cancels against no-TIL control",
}
(ROOT / "results" / "params").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / "results" / "params" / "kill_rate.json", "w"), indent=2)
print(json.dumps(out, indent=2))

fig, ax = plt.subplots(figsize=(4, 3.2))
Rg = np.linspace(0, 4.5, 100)
ax.plot(Rg, 100 * np.exp(-kappa * Rg * 48), color="0.3", label=f"M0 fit, κ = {kappa:.4f} /h")
ax.scatter(R, 100 - df.kill_percent, color="C0", zorder=3, label="paper (48 h)")
ax.scatter(R, 100 - df.kill_loo, marker="x", color="C3", zorder=3, label="leave-one-out prediction")
ax.set(xlabel="TIL : tumor ratio", ylabel="tumor cells surviving (% of control)", ylim=(0, 105))
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "results" / "figures" / "kill_rate.png", dpi=200)
