# Synthetic AR(1) Coupling

# Essential imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import array
import os
import json
import contextlib
import random
import itertools

# imports for DPE
from model.cy_utils import run_causal_analysis
from model.utils import discrete

# imports for TE
from transfer_entropy import compute_te, te_surrogate_threshold, te_identify_causality

# imports for GC
from granger_causality import compute_gc_values, compute_gc_direction

# imports for LZP
from lempel_ziv_penalty import compute_lzp_values, lzp_surrogate_threshold, lzp_identify_causality


def run_full_experiment(data, results_folder='results/experiment-2'):
    """Run DPE, GC, TE, and LZP on every AR(1) trial and save raw results."""
    rows = []
    for phi, content in data.items():
        for trial, (x_values, y_values) in enumerate(zip(content['X'], content['Y'])):
            x_raw = np.asarray(x_values, dtype=np.float32)
            y_raw = np.asarray(y_values, dtype=np.float32)
            x_bin, y_bin = discrete(x_raw, n_bins=2), discrete(y_raw, n_bins=2)
            list_x, list_y = [int(v) for v in x_bin], [int(v) for v in y_bin]
            lzp_x, lzp_y = [v + 1 for v in list_x], [v + 1 for v in list_y]
            dpe_result = run_causal_analysis(x_bin, y_bin)
            verdict_dpe = dpe_result[0]
            gap_dpe = dpe_result[1] if len(dpe_result) > 1 else np.nan
            verdict_gc, xy_gc, yx_gc = compute_gc_direction(x_raw, y_raw)
            verdict_te, gap_te = te_identify_causality(list_x, list_y)
            verdict_lzp, gap_lzp = lzp_identify_causality(lzp_x, lzp_y)
            outcomes = [('DPE', verdict_dpe, gap_dpe),
                        ('GC', verdict_gc, abs(xy_gc-yx_gc)),
                        ('TE', verdict_te, gap_te),
                        ('LZP', verdict_lzp, gap_lzp)]
            for model, direction, gap in outcomes:
                rows.append({'phi': round(float(phi), 2), 'trial': trial,
                             'model': model, 'direction': direction, 'gap': gap})
    os.makedirs(results_folder, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(f'{results_folder}/experiment-2-all_results.csv', index=False)
    return frame

# # run experiment for AR(1) process
# def ar1_experiment(data, results_folder='results/experiment-2'):
#     results = []
    
#     for phi, content in data.items():
        
#         phi_val = round(float(phi), 2)

#         print(r'Processing $\phi$ ', phi_val)

#         driver_trials = content['Y']
#         response_trials = content['X']

#         number_of_trials = len(driver_trials)
#         print(number_of_trials)

#         for i in range(number_of_trials):
#             # Raw Sequences
#             driver_raw = np.array(driver_trials[i], dtype=np.float32)
#             response_raw = np.array(response_trials[i], dtype=np.float32)

#             # Discretization
#             bin_driver = discrete(driver_raw, n_bins=2)
#             bin_response = discrete(response_raw, n_bins=2)

#             # Data conversion for TE
#             list_driver = [int(i) for i in bin_driver]
#             list_response = [int(j) for j in bin_response]

#             # Data conversion for LZP
#             lzp_driver = [int(i) + 1 for i in bin_driver]
#             lzp_response = [int(j) + 1 for j in bin_response]

#             # DPE
#             verdict_dpe, gap_dpe = run_causal_analysis(bin_response, bin_driver)

#             results.append(
#                 {
#                 'phi': phi_val,
#                 'trial': i,
#                 'model': 'DPE',
#                 'direction': verdict_dpe,
#                 'gap': gap_dpe
#                 }
#             )

#             # GC
#             # code updated with the original method from stats.tsa
#             verdict_gc, xy_gc, yx_gc = compute_gc_direction(response_raw, driver_raw)
#             results.append(
#                 {
#                 'phi': phi_val,
#                 'trial': i,
#                 'model': 'GC',
#                 'direction': verdict_gc,
#                 'gap': abs(xy_gc - yx_gc)
#                 }
#             )

#             # TE
#             verdict_te, gap_te = te_identify_causality(list_response, list_driver)
#             results.append(
#                 {
#                 'phi': phi_val,
#                 'trial': i,
#                 'model': 'TE',
#                 'direction': verdict_te,
#                 'gap': gap_te
#                 }
#             )

#             # LZP
#             verdict_lzp, gap_lzp = lzp_identify_causality(lzp_response, lzp_driver)
#             results.append(
#                 {
#                 'phi': phi_val,
#                 'trial': i,
#                 'model': 'LZP',
#                 'direction': verdict_lzp,
#                 'gap': gap_lzp
#                 }
#             )

#     if not os.path.exists(results_folder):
#         os.makedirs(results_folder)

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(f'{results_folder}/experiment-2-all_results.csv')
    
#     return results_df

# # calculating metrics
def calculate_metrics(df: pd.DataFrame):

    df = df.loc[
        :, ~df.columns.str.contains(r'^Unnamed')
    ].copy()

    df['phi'] = df['phi'].astype(str)

    # only fill NaN in the direction column — don't touch numeric columns
    df['direction'] = df['direction'].fillna('Independence')

    accuracy     = []
    distribution = []

    models = df['model'].unique()
    phis = df['phi'].unique()

    for model, phi in itertools.product(models, phis):

        subset = df[
            (df['model'] == model) &
            (df['phi'] == phi)
        ]
        total_trials = len(subset)

        if total_trials == 0:
            continue

        counts = subset['direction'].value_counts()

        xy_count          = int(counts.get('X -> Y',       0))
        yx_count          = int(counts.get('Y -> X',       0))
        independent_count = int(counts.get('Independence', 0))

        accuracy.append({
            'phi':    phi,
            'model':    model,
            'accuracy': round(yx_count / total_trials, 4) if phi != '0.0' else round(independent_count / total_trials, 4),
        })

        distribution.append({
            'phi':             phi,
            'model':             model,
            'total_trials':      total_trials,
            'xy_count':          xy_count,
            'yx_count':          yx_count,
            'independent_count': independent_count,
        })

    accuracy_df     = pd.DataFrame(accuracy)
    distribution_df = pd.DataFrame(distribution)
    return accuracy_df, distribution_df

# # accuracy plot
# def plot_accuracy(df, results_folder='results/experiment-2'):
#     plt.figure(figsize=(9, 7))
#     markers = ['o', 'v', 's', 'p']
    
#     for i, model in enumerate(df['model'].unique()):
#         subset = df[df['model'] == model].sort_values('phi')
#         plt.plot(subset['phi'], subset['accuracy'], 
#                  marker=markers[i % len(markers)], 
#                  label=model, linewidth=2, markersize=12)
    
#     fs = 26
    
#     plt.xlabel(r'Coupling Strength ($\phi$)', fontsize=fs-5, fontweight='bold')
#     plt.ylabel('Accuracy', fontsize=fs-5, fontweight='bold')

#     ax = plt.gca()
#     for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#         label.set_fontsize(fs)

#     leg = plt.legend(
#         bbox_to_anchor=(0.98, 0.80), 
#         loc='center right',
#         frameon=True,
#         framealpha=0.8,
#         fontsize=fs-5
#     )
    

#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f'{results_folder}/accuracy-plot.png')
#     plt.show()

# def bar_plots(df, results_folder='results/experiment-2'):
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     axes = axes.flatten()

#     models = df['model'].unique()
    
#     rename_map = {
#         'xy_count': 'X -> Y', 
#         'yx_count': 'Y -> X', 
#         'independent_count': 'Independent'
#     }

#     for i, model in enumerate(models):
#         subset = df[df['model'] == model].copy()
        
#         subset = subset.rename(columns=rename_map)
        
#         subset.set_index('phi')[['Y -> X', 'X -> Y', 'Independent']].plot(
#             kind='bar', 
#             ax=axes[i], 
#             edgecolor='black',
#             width=0.8
#         )
#         axes[i].set_title(model, fontsize=20)

#         axes[i].set_ylabel('Count', fontsize=18)
#         axes[i].set_xlabel(r'Coupling Strength ($\phi$)', fontsize=18)
        
#         axes[i].tick_params(axis='both', labelsize=18)
#         axes[i].tick_params(axis='x', rotation=0)
        
#         axes[i].legend(fontsize=13, loc='upper right')
        
#         axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
#         for container in axes[i].containers:
#             axes[i].bar_label(container, padding=3, fontsize=12)
            
#         axes[i].margins(y=0.2)

#     plt.tight_layout()
#     plt.savefig(f'{results_folder}/barplots.png')
#     plt.show()

# def wilson_ci(k, n, z=1.96):
#     k   = np.asarray(k, dtype=float)
#     n   = np.asarray(n, dtype=float)
#     p   = k / n
#     z2  = z ** 2

#     center     = (p + z2 / (2 * n)) / (1 + z2 / n)
#     half_width = (z * np.sqrt(p * (1 - p) / n + z2 / (4 * n**2))) / (1 + z2 / n)

#     return np.clip(center - half_width, 0, 1), np.clip(center + half_width, 0, 1)


# def plot_accuracy_with_ci(accuracy_df, distribution_df, results_folder='results/experiment-2'):
#     dist = distribution_df.copy()
#     dist['phi'] = dist['phi'].astype(str)
#     dist['success_count'] = np.where(
#         dist['phi'] == '0.0',
#         dist['independent_count'],
#         dist['yx_count']
#     )

#     plot_df = accuracy_df.merge(
#         dist[['phi', 'model', 'success_count', 'total_trials']],
#         on=['phi', 'model'],
#         how='left'
#     )
#     plot_df['phi_num'] = pd.to_numeric(plot_df['phi'])

#     model_order = ['DPE', 'GC', 'TE', 'LZP']
#     plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)
#     plot_df = plot_df.sort_values(['model', 'phi_num'])

#     markers = ['o', 'v', 'p', 's']   
#     fs      = 26
#     plt.figure(figsize=(9, 7))
#     ax = plt.gca()

#     for i, (model, grp) in enumerate(plot_df.groupby('model', observed=True, sort=True)):
#         grp   = grp.sort_values('phi_num')
#         phis  = grp['phi_num'].values
#         acc   = grp['accuracy'].values
#         lower, upper = wilson_ci(grp['success_count'].values,
#                                  grp['total_trials'].values)
#         color = f'C{i}'
#         ax.fill_between(phis, lower, upper, alpha=0.15, color=color)
#         ax.errorbar(
#             phis, acc,
#             yerr=[acc - lower, upper - acc],
#             fmt=markers[i % len(markers)],
#             color=color,
#             ecolor=color,
#             elinewidth=1.2,
#             capsize=5,
#             capthick=1.5,
#             linewidth=2,
#             markersize=12,
#             label=model
#         )

#     ax.set_xlabel(r'Coupling Strength ($\phi$)', fontsize=fs - 5, fontweight='bold')
#     ax.set_ylabel('Accuracy', fontsize=fs - 5, fontweight='bold')
#     phi_ticks = sorted(plot_df['phi_num'].unique())
#     ax.set_xticks(phi_ticks)
#     ax.set_xticklabels([str(p) for p in phi_ticks])
#     ax.set_ylim(-0.02, 1.05)

#     for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#         label.set_fontsize(fs)

#     ax.legend(
#         loc='center right',
#         frameon=True,
#         framealpha=0.8,
#         fontsize=fs - 5
#     )
#     ax.annotate(
#         'Shaded bands + errorbars = Wilson 95% CI',
#         xy=(0.02, 0.02), xycoords='axes fraction',
#         fontsize=fs - 12, color='grey', style='italic'
#     )
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     os.makedirs(results_folder, exist_ok=True)
#     plt.savefig(f'{results_folder}/ar1-accuracy-with-wilson-ci.png', dpi=150)
#     plt.show()


PAPER_RC = {
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'lines.linewidth': 1.0, 'lines.markersize': 4,
    'axes.linewidth': 0.6, 'grid.linewidth': 0.4,
    'pdf.fonttype': 42, 'ps.fonttype': 42,      # embed fonts (no Type-3)
    'savefig.dpi': 600,
}
plt.rcParams.update(PAPER_RC)
 
def wilson_ci(k, n, z=1.96):
    k = np.asarray(k, dtype=float); n = np.asarray(n, dtype=float)
    p = k / n; z2 = z ** 2
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = (z * np.sqrt(p * (1 - p) / n + z2 / (4 * n**2))) / (1 + z2 / n)
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)
 
def bar_plots(df, results_folder='results/experiment-2'):
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.9))       # full text width
    axes = axes.flatten()
    rename_map = {'xy_count': 'X -> Y', 'yx_count': 'Y -> X', 'independent_count': 'Independent'}
    for i, model in enumerate(df['model'].unique()):
        subset = df[df['model'] == model].copy().rename(columns=rename_map)
        subset.set_index('phi')[['Y -> X', 'X -> Y', 'Independent']].plot(
            kind='bar', ax=axes[i], edgecolor='black', width=0.8, linewidth=0.4)
        axes[i].set_title(model, fontsize=25, fontweight='bold')
        axes[i].set_ylabel('Count', fontsize=23)
        axes[i].set_xlabel(r'Coupling Strength ($\phi$)', fontsize=23)
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].legend(loc='upper center', ncol=3, fontsize=6, frameon=False,
                       columnspacing=0.8, handlelength=1.0, handletextpad=0.4)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].margins(y=0.25)
        # NOTE: per-bar count labels (bar_label) removed: 30 bars in a ~2.3 in panel
        # cannot carry >=6 pt numbers. Counts remain in experiment-2-proportions-count.csv.
    fig.tight_layout(pad=0.4)
    os.makedirs(results_folder, exist_ok=True)
    fig.savefig(f'{results_folder}/ar-1barplots.pdf')        # no bbox_inches='tight' -> exact 4.8 in
    plt.close(fig)
 
def plot_accuracy_with_ci(accuracy_df, distribution_df, results_folder='results/experiment-2'):
    dist = distribution_df.copy()
    dist['phi'] = dist['phi'].astype(str)
    dist['success_count'] = np.where(dist['phi'] == '0.0', dist['independent_count'], dist['yx_count'])
    plot_df = accuracy_df.merge(dist[['phi', 'model', 'success_count', 'total_trials']],
                                on=['phi', 'model'], how='left')
    plot_df['phi_num'] = pd.to_numeric(plot_df['phi'])
    model_order = ['DPE', 'GC', 'TE', 'LZP']
    plot_df['model'] = pd.Categorical(plot_df['model'], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values(['model', 'phi_num'])
 
    markers = ['o', 'v', 'p', 's']
    fig, ax = plt.subplots(figsize=(2.2, 1.85))               # = 0.45*\textwidth, so scale is 1:1
    for i, (model, grp) in enumerate(plot_df.groupby('model', observed=True, sort=True)):
        grp = grp.sort_values('phi_num')
        phis, acc = grp['phi_num'].values, grp['accuracy'].values
        lower, upper = wilson_ci(grp['success_count'].values, grp['total_trials'].values)
        color = f'C{i}'
        ax.fill_between(phis, lower, upper, alpha=0.15, color=color, linewidth=0)
        ax.errorbar(phis, acc, yerr=[acc - lower, upper - acc],
                    fmt=markers[i % len(markers)], color=color, ecolor=color,
                    elinewidth=0.7, capsize=1.5, capthick=0.7, linewidth=1.0,
                    markersize=3, label=model)
    ax.set_xlabel(r'Coupling Strength ($\phi$)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    phi_ticks = sorted(plot_df['phi_num'].unique())[::2]       # every 2nd tick so labels do not touch
    ax.set_xticks(phi_ticks); ax.set_xticklabels([str(p) for p in phi_ticks])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='center right', frameon=True, framealpha=0.8, fontsize=6,
              handlelength=1.2, borderpad=0.3, labelspacing=0.2)
    # NOTE: the "Shaded bands + errorbars = Wilson 95% CI" annotation is removed: it is
    # wider than the panel at >=6 pt and the paper's figure caption already says this.
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.3)
    os.makedirs(results_folder, exist_ok=True)
    fig.savefig(f'{results_folder}/ar1-accuracy-with-wilson-ci.pdf')
    plt.close(fig)


# # extract json data
# ar_path = '../dataset/ar1_data.json'
# ar_data = extract_json(ar_path)

# # run analysis
# summary_df = ar1_experiment(ar_data)
# print(summary_df)

if __name__ == '__main__':
    with open('../dataset/ar1_data.json') as source:
        experiment_data = json.load(source)
    summary_df = run_full_experiment(experiment_data)
    acc_df, dist_df = calculate_metrics(summary_df)
    acc_df.to_csv('results/experiment-2/experiment-2-accuracy.csv', index=False)
    dist_df.to_csv('results/experiment-2/experiment-2-proportions-count.csv', index=False)
    bar_plots(dist_df)
    plot_accuracy_with_ci(acc_df, dist_df)
