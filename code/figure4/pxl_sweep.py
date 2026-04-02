#!/usr/bin/env python3
"""p_xl sweep at fixed fiber geometry: connectivity (S) and rigidity (G) transitions."""
import numpy as np
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fiber_sim import FiberNetwork

BASE = Path(__file__).parent
OUT = BASE / 'pxl_sweep_results'
FIG_DIR = OUT / 'figures'

BUDGET = 300
L_BOX = 10.0
KAPPA = 1.0
K_SHAPE = 4
GAMMA = 0.01

LAMBDAS = [2.5, 3.5]
PXL_PRIMARY = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
               0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]
N_SEEDS_PRIMARY = 100

PXL_CONTROL = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
N_SEEDS_CONTROL = 50
LAMBDA_CONTROL = 3.5
KAPPA_CONTROL = 0.0

N_WORKERS = 128


def run_one(args):
    lam, p_xl, seed, kappa, k_shape, budget, L_box, gamma = args
    net = FiberNetwork(L=L_box, budget=budget, lam=lam, k_shape=k_shape,
                       p_xl=p_xl, kappa=kappa, seed=seed)
    net.build()

    S = net.giant_component_frac()
    z = net.mean_z()
    n_nodes = net.n_nodes
    n_edges = net.n_edges
    n_comp = net.n_components()
    beta1 = n_edges - n_nodes + n_comp

    k_per_fiber = net.crosslinks_per_fiber()
    k_mean = float(k_per_fiber.mean())
    k_dist = np.bincount(k_per_fiber).tolist()

    G = net.shear_modulus(gamma=gamma)

    return {
        'lambda': float(lam), 'p_xl': float(p_xl), 'seed': int(seed),
        'kappa': float(kappa),
        'S': float(S), 'z': float(z),
        'N_nodes': int(n_nodes), 'N_edges': int(n_edges),
        'N_components': int(n_comp), 'beta1': int(beta1),
        'k_mean': float(k_mean), 'k_distribution': k_dist,
        'G': float(G),
    }


def run_sweep(jobs, desc, n_workers):
    results = []
    cs = max(1, len(jobs) // (n_workers * 4))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for r in tqdm(pool.map(run_one, jobs, chunksize=cs),
                      total=len(jobs), desc=desc):
            results.append(r)
    return results


def run_production(n_workers):
    OUT.mkdir(parents=True, exist_ok=True)

    n_primary = len(LAMBDAS) * len(PXL_PRIMARY) * N_SEEDS_PRIMARY
    print()
    print(f'PRIMARY: {len(LAMBDAS)} lam x {len(PXL_PRIMARY)} p_xl x {N_SEEDS_PRIMARY} seeds = {n_primary}')
    print(f'budget={BUDGET}, L={L_BOX}, kappa={KAPPA}, k_shape={K_SHAPE}, gamma={GAMMA}')
    print()

    primary_jobs = [
        (lam, p, s, KAPPA, K_SHAPE, BUDGET, L_BOX, GAMMA)
        for lam in LAMBDAS for p in PXL_PRIMARY for s in range(N_SEEDS_PRIMARY)
    ]
    primary = run_sweep(primary_jobs, 'Primary', n_workers)

    with open(OUT / 'primary_results.json', 'w') as f:
        json.dump(primary, f)
    print(f'  Saved {len(primary)} results')

    n_ctrl = len(PXL_CONTROL) * N_SEEDS_CONTROL
    print()
    print(f'CONTROL (kappa=0): {len(PXL_CONTROL)} p_xl x {N_SEEDS_CONTROL} seeds = {n_ctrl}')
    print()

    ctrl_jobs = [
        (LAMBDA_CONTROL, p, s, KAPPA_CONTROL, K_SHAPE, BUDGET, L_BOX, GAMMA)
        for p in PXL_CONTROL for s in range(N_SEEDS_CONTROL)
    ]
    control = run_sweep(ctrl_jobs, 'Control kappa=0', n_workers)

    with open(OUT / 'control_results.json', 'w') as f:
        json.dump(control, f)
    print(f'  Saved {len(control)} results')

    return primary, control


def _find_threshold(pxl, vals, thresh):
    """Interpolate p_xl where vals first crosses thresh."""
    for i in range(len(pxl) - 1):
        if vals[i] < thresh <= vals[i + 1]:
            f = (thresh - vals[i]) / (vals[i + 1] - vals[i])
            return pxl[i] + f * (pxl[i + 1] - pxl[i])
    return None


def make_figures(primary=None, control=None):
    import pandas as pd

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if primary is None:
        with open(OUT / 'primary_results.json') as f:
            primary = json.load(f)
    if control is None:
        with open(OUT / 'control_results.json') as f:
            control = json.load(f)

    df = pd.DataFrame(primary)
    df_ctrl = pd.DataFrame(control)

    # Print detailed table
    print(f'\n{"=" * 80}')
    print(f'{"lam":>5} {"p_xl":>5} {"S":>7} {"G_med":>10} {"G_mean":>10} {"G_max":>10} '
          f'{"<k>":>6} {"z":>5} {"N":>5} {"beta1":>6}')
    print()
    for lam in sorted(df['lambda'].unique()):
        for pxl in sorted(df[df['lambda'] == lam]['p_xl'].unique()):
            sub = df[(df['lambda'] == lam) & (df['p_xl'] == pxl)]
            print(f'{lam:5.1f} {pxl:5.2f} {sub["S"].mean():7.3f} '
                  f'{sub["G"].median():10.4f} {sub["G"].mean():10.4f} '
                  f'{sub["G"].max():10.2f} {sub["k_mean"].mean():6.1f} '
                  f'{sub["z"].mean():5.2f} {sub["N_nodes"].mean():5.0f} '
                  f'{sub["beta1"].mean():6.0f}')
    print()

    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'axes.linewidth': 1.5, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'savefig.dpi': 400, 'figure.dpi': 150, 'font.family': 'sans-serif',
        'axes.spines.top': False,
    })

    c_S, c_G, c_k = '#2166AC', '#B2182B', '#E6550D'

    # Figure 1: S and G/G_max vs p_xl (both lambdas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, lam in zip(axes, LAMBDAS):
        sub = df[df['lambda'] == lam]
        agg = sub.groupby('p_xl').agg(
            S_mean=('S', 'mean'), S_std=('S', 'std'),
            G_median=('G', 'median'),
            G_q25=('G', lambda x: np.percentile(x, 25)),
            G_q75=('G', lambda x: np.percentile(x, 75)))
        pxl = agg.index.values
        S_m, S_s = agg['S_mean'].values, agg['S_std'].values
        G_m = agg['G_median'].values
        G_lo = agg['G_q25'].values
        G_hi = agg['G_q75'].values

        G_max = max(G_m.max(), 1e-10)
        Gn = G_m / G_max
        Gn_lo = G_lo / G_max
        Gn_hi = G_hi / G_max

        p_conn = _find_threshold(pxl, S_m, 0.5)
        p_rigid = _find_threshold(pxl, Gn, 0.1)

        if p_conn is not None and p_rigid is not None and p_rigid > p_conn:
            ax.axvspan(p_conn, p_rigid, alpha=0.10, color='gold', zorder=0,
                       label='Connected but floppy')

        ax.fill_between(pxl, np.clip(S_m - S_s, 0, 1),
                        np.clip(S_m + S_s, 0, 1), alpha=0.15, color=c_S)
        ax.plot(pxl, S_m, '-o', color=c_S, ms=5, lw=2, label='S')

        ax2 = ax.twinx()
        ax2.fill_between(pxl, np.clip(Gn_lo, 0, 1.3),
                         np.clip(Gn_hi, 0, 1.3), alpha=0.15, color=c_G)
        ax2.plot(pxl, np.clip(Gn, 0, None), '-s', color=c_G, ms=5, lw=2,
                 label=r'G/G$_{\mathrm{max}}$')
        ax2.set_ylabel(r'G / G$_{\mathrm{max}}$', color=c_G)
        ax2.tick_params(axis='y', labelcolor=c_G)
        ax2.set_ylim(-0.05, 1.15)
        ax2.spines['top'].set_visible(False)

        ax.set_xlabel(r'Crosslink fraction $p_{\mathrm{xl}}$')
        ax.set_ylabel('S (giant component)', color=c_S)
        ax.tick_params(axis='y', labelcolor=c_S)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(r'$\lambda$ = ' + f'{lam}')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

        if p_conn is not None:
            ax.axvline(p_conn, color=c_S, ls=':', alpha=0.5, lw=1)
        if p_rigid is not None:
            ax.axvline(p_rigid, color=c_G, ls=':', alpha=0.5, lw=1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_pxl_SG.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig_pxl_SG.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig_pxl_SG')

    # Figure 2: <k> vs p_xl with Maxwell threshold
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors_lam = [c_k, c_S]

    for lam, clr, mkr in zip(LAMBDAS, colors_lam, ['o', 's']):
        sub = df[df['lambda'] == lam]
        agg = sub.groupby('p_xl').agg({'k_mean': ['mean', 'std'], 'G': 'mean'})
        pxl = agg.index.values
        k_m = agg[('k_mean', 'mean')].values
        k_s = agg[('k_mean', 'std')].values
        G_m = agg[('G', 'mean')].values

        ax.fill_between(pxl, k_m - k_s, k_m + k_s, alpha=0.12, color=clr)
        ax.plot(pxl, k_m, f'-{mkr}', ms=5, lw=2, color=clr,
                label=r'$\lambda$=' + f'{lam}')

        G_max = max(G_m.max(), 1e-10)
        Gn = G_m / G_max
        p_rigid = _find_threshold(pxl, Gn, 0.1)
        if p_rigid is not None:
            idx = np.argmin(np.abs(pxl - p_rigid))
            ax.plot(pxl[idx], k_m[idx], '*', color=c_G, ms=15, zorder=5)

    ax.axhline(3.0, color='k', ls='--', alpha=0.5, lw=1.5,
               label=r'$\langle k \rangle$ = 3 (Maxwell-Calladine)')
    ax.set_xlabel(r'Crosslink fraction $p_{\mathrm{xl}}$')
    ax.set_ylabel(r'Mean crosslinks per fiber $\langle k \rangle$')
    ax.legend(fontsize=10)
    ax.set_title(r'Rigidity onset at $\langle k \rangle \approx 3$')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_pxl_k_mean.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig_pxl_k_mean.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig_pxl_k_mean')

    # Figure 3: kappa=0 control
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # kappa=1 reference (use same lambda as control)
    sub = df[df['lambda'] == LAMBDA_CONTROL]
    agg_k1 = sub.groupby('p_xl').agg(S_mean=('S', 'mean'), G_median=('G', 'median'))
    pxl_k1 = agg_k1.index.values
    G_max_k1 = max(agg_k1['G_median'].values.max(), 1e-10)

    ax.plot(pxl_k1, agg_k1['S_mean'].values, '--', color=c_S, lw=1.5, alpha=0.5,
            label=r'S ($\kappa$=1)')
    ax.plot(pxl_k1, np.clip(agg_k1['G_median'].values / G_max_k1, 0, None),
            '-s', color=c_G, ms=4, lw=2,
            label=r'G/G$_{\mathrm{max}}$ ($\kappa$=1)')

    # kappa=0
    agg_k0 = df_ctrl.groupby('p_xl').agg(S_mean=('S', 'mean'), G_median=('G', 'median'))
    pxl_k0 = agg_k0.index.values

    ax.plot(pxl_k0, agg_k0['S_mean'].values, '--', color='#999999', lw=1.5, alpha=0.5,
            label=r'S ($\kappa$=0)')
    # normalize kappa=0 G by kappa=1 G_max so they're comparable
    ax.plot(pxl_k0, np.clip(agg_k0['G_median'].values / G_max_k1, 0, None),
            '-o', color='#999999', ms=4, lw=2,
            label=r'G/G$_{\mathrm{max}}$ ($\kappa$=0)')

    ax.set_xlabel(r'Crosslink fraction $p_{\mathrm{xl}}$')
    ax.set_ylabel(r'S,  G/G$_{\mathrm{max}}$')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=10)
    ax.set_title(r'Bending stiffness ($\kappa > 0$) required for rigidity')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_pxl_kappa0_control.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig_pxl_kappa0_control.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig_pxl_kappa0_control')

    # Figure 4: Simulation vs experiment comparison
    exp_time = np.array([5, 60, 95, 115, 125])
    exp_xi = np.array([35, 100, 110, 112, 115])      # um
    exp_work = np.array([4600, 6500, 7900, 19200, 23700])  # kBT

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: simulation (lambda=2.5)
    ax = axes[0]
    sub = df[df['lambda'] == 2.5]
    agg = sub.groupby('p_xl').agg(
        S_mean=('S', 'mean'), S_std=('S', 'std'),
        G_median=('G', 'median'),
        G_q25=('G', lambda x: np.percentile(x, 25)),
        G_q75=('G', lambda x: np.percentile(x, 75)))
    pxl = agg.index.values
    S_m, S_s = agg['S_mean'].values, agg['S_std'].values
    G_m = agg['G_median'].values
    G_max = max(G_m.max(), 1e-10)
    Gn = G_m / G_max
    Gn_lo = agg['G_q25'].values / G_max
    Gn_hi = agg['G_q75'].values / G_max

    p_conn = _find_threshold(pxl, S_m, 0.5)
    p_rigid = _find_threshold(pxl, Gn, 0.1)
    if p_conn is not None and p_rigid is not None and p_rigid > p_conn:
        ax.axvspan(p_conn, p_rigid, alpha=0.10, color='gold', zorder=0)

    ax.fill_between(pxl, np.clip(S_m - S_s, 0, 1),
                    np.clip(S_m + S_s, 0, 1), alpha=0.15, color=c_S)
    ax.plot(pxl, S_m, '-o', color=c_S, ms=5, lw=2, label='S')
    ax2 = ax.twinx()
    ax2.fill_between(pxl, np.clip(Gn_lo, 0, 1.3),
                     np.clip(Gn_hi, 0, 1.3), alpha=0.15, color=c_G)
    ax2.plot(pxl, np.clip(Gn, 0, None), '-s', color=c_G, ms=5, lw=2,
             label=r'G/G$_{\mathrm{max}}$')
    ax.set_xlabel(r'Crosslink fraction $p_{\mathrm{xl}}$')
    ax.set_ylabel('S', color=c_S)
    ax2.set_ylabel(r'G/G$_{\mathrm{max}}$', color=c_G)
    ax.set_ylim(-0.05, 1.15)
    ax2.set_ylim(-0.05, 1.15)
    ax2.spines['top'].set_visible(False)
    ax.set_title(r'A. Simulation ($\lambda$ = 2.5)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    # Panel B: experiment
    ax = axes[1]
    xi_n = exp_xi / exp_xi.max()
    w_n = exp_work / exp_work.max()

    ax.plot(exp_time, xi_n, '-o', color=c_S, ms=8, lw=2,
            label=r'$\xi / \xi_{\mathrm{max}}$')
    ax2 = ax.twinx()
    ax2.plot(exp_time, w_n, '-s', color=c_G, ms=8, lw=2,
             label=r'$W / W_{\mathrm{max}}$')
    ax.axvspan(35, 95, alpha=0.10, color='gold', zorder=0)
    ax.set_xlabel('Incubation time (min)')
    ax.set_ylabel(r'$\xi / \xi_{\mathrm{max}}$', color=c_S)
    ax2.set_ylabel(r'$W / W_{\mathrm{max}}$', color=c_G)
    ax.set_ylim(-0.05, 1.15)
    ax2.set_ylim(-0.05, 1.15)
    ax2.spines['top'].set_visible(False)
    ax.set_title('B. Experiment')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_sim_vs_exp.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig_sim_vs_exp.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig_sim_vs_exp')

    #Summary    print()
    print('SUMMARY')
    print()
    for lam in LAMBDAS:
        sub = df[df['lambda'] == lam]
        agg = sub.groupby('p_xl').agg(
            S_mean=('S', 'mean'), G_median=('G', 'median'),
            k_mean=('k_mean', 'mean'), z_mean=('z', 'mean'))
        pxl = agg.index.values
        S, G, k, z = (agg['S_mean'].values, agg['G_median'].values,
                       agg['k_mean'].values, agg['z_mean'].values)
        G_max = max(G.max(), 1e-10)
        Gn = G / G_max

        p_conn = _find_threshold(pxl, S, 0.5)
        p_rigid = _find_threshold(pxl, Gn, 0.1)
        gap = (p_rigid - p_conn) if p_conn and p_rigid else None

        print(f'\n  lambda = {lam}:')
        print(f'    p_xl_conn  (S>0.5)      = {p_conn:.3f}' if p_conn else '    p_xl_conn: not found')
        print(f'    p_xl_rigid (G/Gmax>0.1) = {p_rigid:.3f}' if p_rigid else '    p_xl_rigid: not found')
        print(f'    gap = {gap:.3f}' if gap else '    gap: N/A')
        print(f'    G_max = {G_max:.6f}')
        print(f'    z range: {z.min():.2f} -- {z.max():.2f}')
        print(f'    <k> range: {k.min():.1f} -- {k.max():.1f}')
        if p_rigid is not None:
            idx = np.argmin(np.abs(pxl - p_rigid))
            print(f'    <k> at rigidity: {k[idx]:.2f}')

    print(f'\n  kappa=0 control (lambda={LAMBDA_CONTROL}):')
    print(f'    max G = {df_ctrl["G"].max():.6f} (should be ~0)')
    print()
    print(f'\nFigures saved to: {FIG_DIR}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--figures-only', action='store_true')
    p.add_argument('--n-workers', type=int, default=N_WORKERS)
    args = p.parse_args()

    if args.figures_only:
        make_figures()
        return

    primary, control = run_production(args.n_workers)
    make_figures(primary, control)


if __name__ == '__main__':
    main()
