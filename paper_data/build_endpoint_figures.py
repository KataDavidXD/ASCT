"""Reproduce publication figures from retained, matched seed-level records.

Run with matplotlib, numpy, Pillow, and pypdf installed. No training is run.
Outputs: vector PDF/SVG, 300-dpi PNG, plot data, and source/hash checks.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
SEEDS = (11, 23, 37)
METHODS = ("PPO", "VinePPO", "Uniform", "UCT", "AgentUCT")
NAMES = {"Base": "Base", "PPO": "PPO", "VinePPO": "VinePPO",
         "Uniform": "ASCT-Uniform", "UCT": "ASCT-UCT", "AgentUCT": "ASCT-AgentUCT"}
COLORS = {"Base": "#999999", "PPO": "#555555", "VinePPO": "#9B6A1C",
          "Uniform": "#737A35", "UCT": "#936490", "AgentUCT": "#16577C"}
STYLES = {"PPO": (0, (4, 2)), "VinePPO": (0, (5, 2, 1, 2)),
          "Uniform": (0, (2, 1.5)), "UCT": (0, (6, 2)), "AgentUCT": "-"}
MARKERS = {"PPO": "s", "VinePPO": "^", "Uniform": "o", "UCT": "v", "AgentUCT": "D"}
BETA = 1e-4
FILES = ["results/experiment-1-main/endpoints.csv",
         "results/experiment-3-meta-token/search_costs.csv",
         "results/paper-metrics-v14/deployment_by_seed.csv",
         "results/paper-metrics-v14/deployment_crossings.csv"]


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


endpoint_rows, cost_rows, stored_j, crossings = map(read_csv, FILES)
selected = [r for r in endpoint_rows if r["setting"] == "matched" and r["split"] == "test"]
assert len(selected) == 16
for m in METHODS:
    rs = [r for r in selected if r["method"] == m]
    assert len(rs) == 3 and {int(r["seed"]) for r in rs} == set(SEEDS)
    assert all(r["iteration"] == "3" and r["n"] == "400" for r in rs)
for r in selected:
    reconstructed = float(r["f1_official"]) - 0.1 * float(r["execution_words"]) / 4096
    assert abs(reconstructed - float(r["utility_official"])) < 1e-12
ep = {(r["method"], int(r["seed"])): float(r["utility_official"])
      for r in selected if r["method"] != "Base"}
tokens = {(r["method"], int(r["seed"])): int(r["qwen_tokens"])+int(r['auxiliary_actor_scoring_tokens']) for r in cost_rows}
assert len(ep) == len(tokens) == 15
for r in cost_rows:
    assert int(r["qwen_tokens"]) == sum(int(r[k]) for k in (
        "generator_4b_input_tokens", "generator_4b_output_tokens",
        "query_embedding_06b_input_tokens", "reranker_06b_input_tokens"))
u = {m: np.array([ep[m, seed] for seed in SEEDS]) for m in METHODS}
t = {m: np.array([tokens[m, seed] for seed in SEEDS]) for m in METHODS}
base = float(next(r["utility_official"] for r in selected if r["method"] == "Base"))


def j_by_seed(method, n):
    return u[method][:, None] - BETA * t[method][:, None] / np.atleast_1d(n)[None, :]


for r in stored_j:
    val = ep[r["method"], int(r["seed"])] - BETA * tokens[r["method"], int(r["seed"])] / int(r["N"])
    assert abs(val - float(r["j_aux"])) < 1e-12
crossing_values = {}
for r in crossings:
    if r['ledger']!='aux' or not r['positive_crossing_N']:
        continue
    m, c = r["method"], r["control"]
    n_cross = BETA * (t[m].mean() - t[c].mean()) / (u[m].mean() - u[c].mean())
    assert abs(n_cross - float(r["positive_crossing_N"])) < 1e-6
    if m == "AgentUCT":
        crossing_values[c] = float(n_cross)

font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.65,
    "axes.edgecolor": "#555555", "text.color": "#202020",
    "axes.labelcolor": "#202020", "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def clean(ax, axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.6, length=3)


def save(fig, name):
    # The PDF canvas is exactly the manuscript's 5.5-inch text width.
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300,
                    metadata={"Creator": "Matplotlib; retained experiment records"}
                    if ext == "pdf" else None)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1, (name, artist.get_text(), box)
        assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, (name, artist.get_text(), box)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(5.5, 2.24))
fig.subplots_adjust(left=0.255, right=0.887, top=0.91, bottom=0.22)
ordered = ("Base", *METHODS)
for y, m in enumerate(ordered):
    values = np.array([base]) if m == "Base" else u[m]
    avg = values.mean()
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    # Full-height dashed mean guides; muted so data and SD intervals stay primary.
    ax.axvline(avg, color=COLORS[m], linestyle=(0, (4, 3)),
               linewidth=1.0 if m == "AgentUCT" else 0.75,
               alpha=0.55 if m == "AgentUCT" else 0.40, zorder=2)
    if len(values) > 1:
        ax.scatter(values, y + np.array([-0.10, 0, 0.10]), marker="o", s=12,
                   facecolor="white", edgecolor=COLORS[m], linewidth=0.8, zorder=4)
        ax.errorbar(avg, y, xerr=sd, fmt="none", ecolor=COLORS[m],
                    elinewidth=1.7 if m == "AgentUCT" else 1.1,
                    capsize=3, capthick=1.1, zorder=3)
    ax.scatter([avg], [y], marker="D", s=31 if m == "AgentUCT" else 22,
               facecolor=COLORS[m], edgecolor="white", linewidth=0.55, zorder=5)
    ax.text(1.055, y, f"{avg:.4f}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8.8,
            fontweight="bold" if m == "AgentUCT" else "normal", color=COLORS[m])
ax.text(1.055, -0.84, "Mean", transform=ax.get_yaxis_transform(),
        ha="left", va="center", fontsize=8.8, color="#333333")
ax.set_yticks(range(len(ordered)), [NAMES[m] for m in ordered])
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_ylim(5.6, -0.6)
ax.set_xlim(0.54, 0.65)
ax.set_xticks(np.arange(0.54, 0.651, 0.02))
ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.set_xlabel(r"Mean test workflow utility, $\overline{U}_{\mathrm{test}}$", labelpad=4)
ax.tick_params(axis="y", length=0, pad=7)
clean(ax)
ax.grid(False, axis="x")
save(fig, "utility_errorbars")

# Compute each seed's entire analytic projection before mean/SD aggregation.
# This preserves the utility-token covariance and does not invent new runs.
n_values = np.geomspace(100_000, 3_000_000, 501)
fig, ax = plt.subplots(figsize=(5.5, 2.80))
fig.subplots_adjust(left=0.12, right=0.97, top=0.745, bottom=0.185)
handles = []
for m in METHODS:
    jj = j_by_seed(m, n_values)
    avg, sd = jj.mean(axis=0), jj.std(axis=0, ddof=1)
    ax.fill_between(n_values, avg - sd, avg + sd, color=COLORS[m], alpha=0.085, linewidth=0, zorder=1)
    line, = ax.plot(n_values, avg, label=NAMES[m], color=COLORS[m], linestyle=STYLES[m],
                    linewidth=2.1 if m == "AgentUCT" else 1.15,
                    marker=MARKERS[m], markersize=3.1, markevery=[0, 135, 237, 339, 500],
                    markeredgewidth=0.65, markerfacecolor="white" if m != "AgentUCT" else COLORS[m], zorder=4 if m == "AgentUCT" else 3)
    handles.append(line)
ax.set_xscale("log")
ax.set_xlim(n_values[0], n_values[-1])
ax.set_ylim(0.18, 0.645)
ax.set_yticks([0.20, 0.30, 0.40, 0.50, 0.60])
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ticks = [100_000, 250_000, 500_000, 1_000_000, 3_000_000]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.set_xticklabels(["100k", "250k", "500k", "1M", "3M"])
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel(r"Assumed policy uses, $N$ (log scale)", labelpad=4)
ax.set_ylabel(r"Amortized policy value, $J_{\mathrm{deploy}}(N)$", labelpad=5)
clean(ax, "y")
legend = fig.legend(handles, [NAMES[m] for m in METHODS], loc="upper center", ncol=3,
                    bbox_to_anchor=(0.55, 1.005), frameon=False, handlelength=2.3,
                    columnspacing=1.0, handletextpad=0.5, labelspacing=0.65)
for label in legend.get_texts():
    if label.get_text() == "ASCT-AgentUCT":
        label.set_fontweight("bold")
# Only positive crossings exist on the declared total-auxiliary ledger.
for control, text_y, align in [("PPO", 0.33, "left")]:
    nn = crossing_values[control]
    yy = float(j_by_seed("AgentUCT", nn).mean())
    ax.plot([nn, nn], [ax.get_ylim()[0], yy], color="#666666",
            linestyle=(0, (4, 2)), linewidth=0.75, zorder=2)
    ax.scatter([nn], [yy], s=15, facecolor="white", edgecolor=COLORS["AgentUCT"], linewidth=0.85, zorder=6)
    ax.text(nn * (0.95 if align == "right" else 1.06), text_y,
            f"{nn / 1000:.1f}k\nvs. {control}", ha=align, va="center", fontsize=8,
            linespacing=1.0, color="#444444")
save(fig, "amortized_j_curves")

with (OUT / "utility_plot_data.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["method", "seed", "test_utility", "method_mean", "sample_sd", "n_seeds"])
    for m in ordered:
        values = np.array([base]) if m == "Base" else u[m]
        for index, val in enumerate(values):
            writer.writerow([NAMES[m], "" if m == "Base" else SEEDS[index], val, values.mean(),
                             "" if m == "Base" else values.std(ddof=1), len(values)])
with (OUT / "amortization_plot_data.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["method", "N", "seed_11_J", "seed_23_J", "seed_37_J", "mean", "sample_sd"])
    for m in METHODS:
        j = j_by_seed(m, n_values)
        for i, nn in enumerate(n_values):
            writer.writerow([NAMES[m], nn, *j[:, i], j[:, i].mean(), j[:, i].std(ddof=1)])
audit = {"renderer": "Matplotlib", "version": matplotlib.__version__, "font": font_path,
         "sources": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES},
         "filter": "setting=matched; split=test; iteration=3; seeds=11,23,37; n=400 per seed",
         "uncertainty": "Sample standard deviation across training seeds (ddof=1), not a confidence interval",
         "projection": "J_m,r(N) = U_test_m,r - beta * (T_environment + T_auxiliary_actor)_m,r / N; beta=1e-4",
         "N_range": [100_000, 3_000_000], "crossings_ASCT_AgentUCT": crossing_values,
         "checks": {"utility_formula": True, "token_sum": True, "stored_J_all_rows": True,
                    "stored_crossings": True, "text_within_canvas": True},
         "utility": {NAMES[m]: {"mean": float(u[m].mean()), "sd": float(u[m].std(ddof=1))} for m in METHODS}}
(OUT / "figure_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
