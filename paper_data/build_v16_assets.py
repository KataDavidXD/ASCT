"""Build Table 4 and Figure 3 from retained, reconciled training ledgers."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as stats
import numpy as np
import matplotlib
from plot_style import paper_font
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
(HERE / 'figures').mkdir(parents=True, exist_ok=True)
METHODS = ('VinePPO', 'Uniform', 'UCT', 'AgentUCT')
PLOT_METHODS = ('Uniform', 'UCT', 'AgentUCT')
SEEDS = (11, 23, 37)
BETA = 1e-4
NAMES = {m: m if m == 'VinePPO' else 'ASCT-' + m for m in METHODS}
COLORS = {'VinePPO': '#9B6A1C', 'Uniform': '#737A35', 'UCT': '#936490', 'AgentUCT': '#16577C'}


def read(name):
    with (HERE / 'data' / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    ref = json.loads((HERE / 'v16_reference.json').read_text())
    for name, digest in ref['data_hashes'].items():
        assert hashlib.sha256((HERE / 'data' / name).read_bytes()).hexdigest() == digest
    rows = read('training_iterations.csv')
    full = {(r['method'], int(r['seed'])): r for r in read('cost_by_seed.csv')}
    published = {r['method']: r for r in read('cost_summary.csv')}
    summaries, curves = [], []
    lookup = {(r['method'], int(r['seed']), int(r['iteration'])): r for r in rows}
    assert len(lookup) == len(rows)
    for m in METHODS:
        seed_values = []
        for seed in SEEDS:
            rr = [lookup[m, seed, i] for i in (1, 2, 3)]
            assert all(int(r['pulls']) == 12 * int(r['searches']) and int(r['train_qids']) == 2000 for r in rr)
            pulls = sum(int(r['pulls']) for r in rr)
            env = sum(int(r['search_qwen_tokens']) for r in rr)
            actor = sum(int(r['auxiliary_actor_scoring_tokens']) for r in rr)
            utility = sum(float(r['rollout_utility']) * int(r['pulls']) for r in rr) / pulls
            j = utility - BETA * (env + actor) / pulls
            f = full[m, seed]
            assert (pulls, env, actor) == (int(f['pulls']), int(f['environment_tokens']), int(f['auxiliary_actor_tokens']))
            assert math.isclose(utility, float(f['rollout_utility']), abs_tol=1e-10)
            assert math.isclose(j, float(f['j_aux']), abs_tol=1e-10)
            seed_values.append((utility, (env + actor) / 1e6, j))
        record = {'method': m}
        for k, key, source, scale in [(0, 'rollout_u', 'rollout_utility', 1), (1, 'tokens_m', 'auxiliary_total_tokens', 1e6), (2, 'j_search', 'j_aux', 1)]:
            values = [v[k] for v in seed_values]
            record[key + '_mean'] = stats.mean(values)
            record[key + '_sd'] = stats.stdev(values)
            for suffix in ('mean', 'sd'):
                assert math.isclose(record[key + '_' + suffix], float(published[m][source + '_' + suffix]) / scale, abs_tol=1e-9)
        summaries.append(record)
        for i in (1, 2, 3):
            rr = [lookup[m, seed, i] for seed in SEEDS]
            u = [float(r['rollout_utility']) for r in rr]
            j = [float(r['rollout_utility']) - BETA * (int(r['search_qwen_tokens']) + int(r['auxiliary_actor_scoring_tokens'])) / int(r['pulls']) for r in rr]
            curves.append(dict(method=m, iteration=i, rollout_u_mean=stats.mean(u), rollout_u_sd=stats.stdev(u), j_search_mean=stats.mean(j), j_search_sd=stats.stdev(j)))

    for filename, records in [('table4_metrics.csv', summaries), ('training_curve_means.csv', curves)]:
        with (HERE / 'data' / filename).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)

    lines = [r'\begin{table}[!htb]', r'\centering\small', r'\setlength{\tabcolsep}{3pt}',
             r'\caption{Mean auxiliary rollout utility, total auxiliary cost, and $\overline J_{\mathrm{search}}$ over complete training runs. Rollout utility uses the training scorer; tokens include environment execution and auxiliary actor scoring. Values are mean $\pm$ sample SD across three seeds.}',
             r'\label{tab:j_primary}', r'\begin{tabularx}{\linewidth}{@{}Y r r r@{}}', r'\toprule',
             r'Evaluator & Mean rollout $U$ & Auxiliary tokens (M) & $\overline J_{\mathrm{search}}$ \\', r'\midrule']
    for r in summaries:
        cells = [NAMES[r['method']]]
        for key, digits in [('rollout_u', 4), ('tokens_m', 3), ('j_search', 4)]:
            cells.append(f"{r[key + '_mean']:.{digits}f} $\\pm$ {r[key + '_sd']:.{digits}f}")
        if r['method'] != 'VinePPO':
            cells = [r'\textbf{\boldmath ' + c + '}' for c in cells]
        lines.append(' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabularx}', r'\end{table}'])
    (HERE / 'j_comparison.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    plt.rcParams.update({'font.family': 'serif', 'font.serif': [paper_font()], 'font.size': 10,
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05))
    handles = []
    for ax, key, title, ylabel in zip(axes, ('rollout_u', 'j_search'),
        ('(a) Rollout utility', '(b) Cost-adjusted value'),
        (r'Mean rollout $U_{\mathrm{train}}$', r'Mean $J_{\mathrm{search}}$')):
        for m, ls, marker in zip(PLOT_METHODS, ('--', '-.', '-'), ('o', 's', 'D')):
            rr = [r for r in curves if r['method'] == m]
            avg = np.array([r[key + '_mean'] for r in rr]); sd = np.array([r[key + '_sd'] for r in rr])
            h, = ax.plot([1, 2, 3], avg, color=COLORS[m], ls=ls, marker=marker, ms=3.8, lw=1.4)
            ax.fill_between([1, 2, 3], avg-sd, avg+sd, color=COLORS[m], alpha=.13, linewidth=0)
            ax.errorbar([1, 2, 3], avg, yerr=sd, fmt='none', color=COLORS[m], capsize=3, elinewidth=.95, capthick=.95, zorder=4)
            if ax is axes[0]: handles.append(h)
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10); ax.set_xlabel('Training iteration', fontsize=10)
        ax.set_xticks([1, 2, 3]); ax.set_xlim(.94, 3.06)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.16); ax.set_axisbelow(True)
    axes[0].set_ylim(.51, .58); axes[1].set_ylim(.477, .543)
    leg = fig.legend(handles, [NAMES[m] for m in PLOT_METHODS], loc='upper center', ncol=3, frameon=False,
                     bbox_to_anchor=(.50, 1.02), fontsize=10, handlelength=2, columnspacing=1.1)
    leg.get_texts()[-1].set_fontweight('bold')
    fig.subplots_adjust(left=.115, right=.98, bottom=.25, top=.73, wspace=.45)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text(): continue
        b=t.get_window_extent(renderer)
        assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1, t.get_text()
    for ext in ('pdf', 'png'):
        fig.savefig(HERE / 'figures' / ('full_training_utility_value.' + ext), dpi=220)
    plt.close(fig)
    report = dict(status='passed', methods=list(METHODS), plot_methods=list(PLOT_METHODS), seeds=list(SEEDS), iteration_records=36,
                  full_run_aggregation='pull-weighted utility per seed, then mean and sample SD across seeds',
                  curve_aggregation='per-iteration utility and total-cost J per seed, then mean and sample SD',
                  reconciliation='all 12 runs agree with retained full-run utility, pulls, environment tokens and auxiliary actor tokens',
                  changed_figure_sha256=hashlib.sha256((HERE/'figures/full_training_utility_value.pdf').read_bytes()).hexdigest())
    (HERE / 'metrics_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
