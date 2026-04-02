#!/usr/bin/env python3
"""Dimensionless parameter sweep for fiber intersection network.

All lengths in units of mean fiber length lambda.
All energies in units of stretching modulus mu (=1).

Sweeps:
  1. Density (a_tilde) x p_xl          — primary
  2. System size (L_tilde) at fixed a   — finite-size scaling
  3. Bending stiffness (lb_tilde) x p_xl — kappa dependence
  4. Lambda check at fixed dimensionless — consistency
  5. k-collapse from sweep 1 data       — analysis only
"""
import numpy as np
import json, os, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from fiber_sim import FiberNetwork

BASE = Path(__file__).parent / 'dimless_results'

# Fixed
MU = 1.0
K_SHAPE = 4
GAMMA = 0.01
N_WORKERS = 128


def dimless_to_sim(a_tilde, L_tilde, lb_tilde, lam_sim=1.0):
    """Map dimensionless params to simulation inputs."""
    L_sim = L_tilde * lam_sim
    budget_sim = a_tilde * L_sim**2 / lam_sim
    kappa_sim = (lb_tilde * lam_sim)**2 * MU
    return L_sim, budget_sim, kappa_sim


def run_one(args):
    a_tilde, L_tilde, lb_tilde, p_xl, lam_sim, seed = args
    L_sim, budget, kappa = dimless_to_sim(a_tilde, L_tilde, lb_tilde, lam_sim)

    net = FiberNetwork(L=L_sim, budget=budget, lam=lam_sim,
                       k_shape=K_SHAPE, p_xl=p_xl, kappa=kappa, seed=seed)
    net.build()
    m = net.full_measure(GAMMA)

    G_tilde = m['G'] * lam_sim / MU

    return {
        'params': {
            'a_tilde': float(a_tilde), 'L_tilde': float(L_tilde),
            'lb_tilde': float(lb_tilde), 'p_xl': float(p_xl),
            'lam_sim': float(lam_sim), 'L_sim': float(L_sim),
            'budget_sim': float(budget), 'kappa_sim': float(kappa),
            'seed': int(seed),
        },
        'topology': {
            'S': m['S'], 'N_nodes': m['N_nodes'], 'N_edges': m['N_edges'],
            'N_components': m['N_components'], 'beta1': m['beta1'],
            'z_mean': m['z'],
        },
        'per_fiber': {
            'k_mean': m['k_mean'], 'k_std': m['k_std'],
            'k_distribution': m['k_distribution'],
            'N_fibers_total': m['N_fibers_total'],
            'N_fibers_with_crosslinks': m['N_fibers_with_xl'],
            'f_isolated': m['f_isolated'],
        },
        'mechanics': {
            'G': m['G'], 'G_tilde': G_tilde,
            'total_energy': m['E_total'],
            'stretch_energy': m['E_stretch'],
            'bend_energy': m['E_bend'],
            'Gamma': m['Gamma'],
        },
    }


def run_sweep(jobs, desc, n_workers=N_WORKERS):
    cs = max(1, len(jobs) // (n_workers * 4))
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for r in tqdm(pool.map(run_one, jobs, chunksize=cs),
                      total=len(jobs), desc=desc):
            results.append(r)
    return results


def save_results(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f)
    print(f'  Saved {len(results)} results -> {path}')


# ============================================================
# Sweep definitions
# ============================================================

def sweep1_density(n_seeds=100, a_values=None, n_workers=N_WORKERS):
    """Density sweep: vary a_tilde and p_xl."""
    if a_values is None:
        a_values = [3, 4, 5, 6, 7, 8, 10, 12, 15]
    pxl_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                  0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                  0.85, 0.90, 0.95, 1.00]
    L_tilde, lb_tilde, lam_sim = 8.0, 0.4, 1.0

    jobs = [(a, L_tilde, lb_tilde, p, lam_sim, s)
            for a in a_values for p in pxl_values for s in range(n_seeds)]

    print(f'Sweep 1 (density): {len(a_values)} a x {len(pxl_values)} p_xl x {n_seeds} seeds = {len(jobs)}')
    results = run_sweep(jobs, 'Sweep1', n_workers)

    out = BASE / 'sweep1_density'
    save_results(results, out / 'all_results.json')
    return results


def sweep2_size(n_seeds=100, n_workers=N_WORKERS):
    """System size scaling at fixed a_tilde=8."""
    L_values = [4, 6, 8, 10, 12, 15]
    pxl_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    a_tilde, lb_tilde, lam_sim = 8.0, 0.4, 1.0

    jobs = [(a_tilde, Lt, lb_tilde, p, lam_sim, s)
            for Lt in L_values for p in pxl_values for s in range(n_seeds)]

    print(f'Sweep 2 (size): {len(L_values)} L x {len(pxl_values)} p_xl x {n_seeds} seeds = {len(jobs)}')
    results = run_sweep(jobs, 'Sweep2', n_workers)

    out = BASE / 'sweep2_size'
    save_results(results, out / 'all_results.json')
    return results


def sweep3_bending(n_seeds=50, n_workers=N_WORKERS):
    """Bending stiffness sweep."""
    lb_values = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0]
    pxl_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    a_tilde, L_tilde, lam_sim = 8.0, 8.0, 1.0

    jobs = [(a_tilde, L_tilde, lb, p, lam_sim, s)
            for lb in lb_values for p in pxl_values for s in range(n_seeds)]

    print(f'Sweep 3 (bending): {len(lb_values)} lb x {len(pxl_values)} p_xl x {n_seeds} seeds = {len(jobs)}')
    results = run_sweep(jobs, 'Sweep3', n_workers)

    out = BASE / 'sweep3_bending'
    save_results(results, out / 'all_results.json')
    return results


def sweep4_lambda_check(n_seeds=50, n_workers=N_WORKERS):
    """Dimensional consistency: vary lam_sim at fixed dimensionless params."""
    lam_values = [0.5, 0.75, 1.0, 1.5, 2.0]
    pxl_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    a_tilde, L_tilde, lb_tilde = 8.0, 8.0, 0.4

    jobs = [(a_tilde, L_tilde, lb_tilde, p, lam, s)
            for lam in lam_values for p in pxl_values for s in range(n_seeds)]

    print(f'Sweep 4 (lambda check): {len(lam_values)} lam x {len(pxl_values)} p_xl x {n_seeds} seeds = {len(jobs)}')
    results = run_sweep(jobs, 'Sweep4', n_workers)

    out = BASE / 'sweep4_lambda_check'
    save_results(results, out / 'all_results.json')
    return results


# ============================================================
# Figures
# ============================================================

def make_figures():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = BASE / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'axes.linewidth': 1.5, 'font.family': 'sans-serif',
        'axes.spines.top': False, 'axes.spines.right': False,
    })

    c_S, c_G = '#2166AC', '#B2182B'

    # --- Sweep 1: density ---
    s1_path = BASE / 'sweep1_density' / 'all_results.json'
    if s1_path.exists():
        with open(s1_path) as f:
            s1 = json.load(f)
        rows = []
        for r in s1:
            rows.append({
                'a_tilde': r['params']['a_tilde'],
                'p_xl': r['params']['p_xl'],
                'seed': r['params']['seed'],
                'S': r['topology']['S'],
                'G_tilde': r['mechanics']['G_tilde'],
                'k_mean': r['per_fiber']['k_mean'],
                'Gamma': r['mechanics']['Gamma'],
            })
        df = pd.DataFrame(rows)

        a_vals = sorted(df['a_tilde'].unique())
        cmap = plt.cm.viridis
        a_norm = plt.Normalize(min(a_vals), max(a_vals))

        # Fig 4B: k-collapse
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        for a in a_vals:
            sub = df[df['a_tilde'] == a]
            agg = sub.groupby('p_xl').agg(
                S_m=('S', 'mean'), k_m=('k_mean', 'mean'),
                G_m=('G_tilde', 'median'))
            col = cmap(a_norm(a))
            ax1.plot(agg['k_m'], agg['S_m'], '-o', ms=4, lw=2, color=col,
                     label=f'a={a:.0f}')
            Gmax = max(agg['G_m'].max(), 1e-10)
            ax2.plot(agg['k_m'], np.clip(agg['G_m'] / Gmax, 0, None),
                     '-o', ms=4, lw=2, color=col, label=f'a={a:.0f}')

        ax1.set_xlabel(r'$\langle k \rangle$')
        ax1.set_ylabel('S')
        ax1.legend(fontsize=8, ncol=2)
        ax2.set_xlabel(r'$\langle k \rangle$')
        ax2.set_ylabel(r'$\tilde{G}/\tilde{G}_{max}$')
        ax2.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / 'fig4B_k_collapse.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / 'fig4B_k_collapse.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('  fig4B_k_collapse')

        # S and G vs p_xl per density
        n_a = len(a_vals)
        ncols = min(4, n_a)
        nrows = (n_a + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False)
        for idx, a in enumerate(a_vals):
            ax = axes[idx // ncols, idx % ncols]
            sub = df[df['a_tilde'] == a]
            agg = sub.groupby('p_xl').agg(
                S_m=('S', 'mean'), S_s=('S', 'std'),
                G_m=('G_tilde', 'median'))
            pxl = agg.index.values
            S_m = agg['S_m'].values
            G_m = agg['G_m'].values
            Gmax = max(G_m.max(), 1e-10)
            ax.plot(pxl, S_m, '-o', color=c_S, ms=4, lw=2, label='S')
            ax.plot(pxl, np.clip(G_m / Gmax, 0, None), '-s', color=c_G, ms=4, lw=2,
                    label=r'$\tilde{G}/\tilde{G}_{max}$')
            ax.set_ylim(-0.05, 1.15)
            ax.set_xlabel(r'$p_{xl}$')
            ax.set_title(f'a={a:.0f}')
            ax.legend(fontsize=9)
        for idx in range(len(a_vals), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / 'sweep1_SG_vs_pxl.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / 'sweep1_SG_vs_pxl.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('  sweep1_SG_vs_pxl')

        # Gamma vs k
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for a in a_vals:
            sub = df[df['a_tilde'] == a]
            agg = sub.groupby('p_xl').agg(k_m=('k_mean', 'mean'), Gam_m=('Gamma', 'median'))
            col = cmap(a_norm(a))
            ax.plot(agg['k_m'], np.maximum(agg['Gam_m'], 1e-6),
                    '-o', ms=4, lw=2, color=col, label=f'a={a:.0f}')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\langle k \rangle$')
        ax.set_ylabel(r'$\Gamma$ (non-affinity)')
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / 'supp_Gamma_vs_k.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / 'supp_Gamma_vs_k.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('  supp_Gamma_vs_k')

        # Threshold table
        print('\n  Threshold summary (Sweep 1):')
        print(f'  {"a":>4} {"k_geo":>6} {"p_conn":>7} {"p_rigid":>8} {"k_conn":>7} {"k_rigid":>8} {"gap_k":>6}')
        for a in a_vals:
            sub = df[df['a_tilde'] == a]
            agg = sub.groupby('p_xl').agg(
                S_m=('S', 'mean'), G_m=('G_tilde', 'median'),
                k_m=('k_mean', 'mean'))
            pxl = agg.index.values
            S_m = agg['S_m'].values
            G_m = agg['G_m'].values
            k_m = agg['k_m'].values
            Gmax = max(G_m.max(), 1e-10)
            Gn = G_m / Gmax

            p_conn = _interp_thresh(pxl, S_m, 0.5)
            p_rigid = _interp_thresh(pxl, Gn, 0.1)
            k_conn = _interp_thresh(k_m, S_m, 0.5) if p_conn else None
            k_rigid = _interp_thresh(k_m, Gn, 0.1) if p_rigid else None
            k_geo = (2/np.pi) * a
            gap = (k_rigid - k_conn) if k_conn is not None and k_rigid is not None else None

            pc_s = f'{p_conn:.3f}' if p_conn else '  N/A'
            pr_s = f'{p_rigid:.3f}' if p_rigid else '   N/A'
            kc_s = f'{k_conn:.2f}' if k_conn else '  N/A'
            kr_s = f'{k_rigid:.2f}' if k_rigid else '   N/A'
            g_s = f'{gap:.2f}' if gap else ' N/A'
            print(f'  {a:4.0f} {k_geo:6.1f} {pc_s:>7} {pr_s:>8} {kc_s:>7} {kr_s:>8} {g_s:>6}')

    # --- Sweep 2: size ---
    s2_path = BASE / 'sweep2_size' / 'all_results.json'
    if s2_path.exists():
        with open(s2_path) as f:
            s2 = json.load(f)
        rows = [{'L_tilde': r['params']['L_tilde'], 'p_xl': r['params']['p_xl'],
                 'S': r['topology']['S'], 'G_tilde': r['mechanics']['G_tilde']}
                for r in s2]
        df2 = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        L_vals = sorted(df2['L_tilde'].unique())
        for Lt in L_vals:
            sub = df2[df2['L_tilde'] == Lt]
            agg = sub.groupby('p_xl').agg(G_m=('G_tilde', 'median'))
            Gmax = max(agg['G_m'].max(), 1e-10)
            ax.plot(agg.index, np.clip(agg['G_m'] / Gmax, 0, None),
                    '-o', ms=4, lw=2, label=f'L={Lt:.0f}')
        ax.set_xlabel(r'$p_{xl}$')
        ax.set_ylabel(r'$\tilde{G}/\tilde{G}_{max}$')
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / 'supp_size_scaling.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / 'supp_size_scaling.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('  supp_size_scaling')

    # --- Sweep 3: bending ---
    s3_path = BASE / 'sweep3_bending' / 'all_results.json'
    if s3_path.exists():
        with open(s3_path) as f:
            s3 = json.load(f)
        rows = [{'lb_tilde': r['params']['lb_tilde'], 'p_xl': r['params']['p_xl'],
                 'S': r['topology']['S'], 'G_tilde': r['mechanics']['G_tilde'],
                 'k_mean': r['per_fiber']['k_mean']}
                for r in s3]
        df3 = pd.DataFrame(rows)

        # Fig 4C: kappa=0 control
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for lb in [0, 0.4]:
            sub = df3[np.isclose(df3['lb_tilde'], lb)]
            if sub.empty:
                continue
            agg = sub.groupby('p_xl').agg(S_m=('S', 'mean'), G_m=('G_tilde', 'median'))
            pxl = agg.index.values
            ls = '-' if lb > 0 else '--'
            c_s = c_S if lb > 0 else '#999'
            c_g = c_G if lb > 0 else '#999'
            tag = f'lb={lb}'
            ax.plot(pxl, agg['S_m'], ls, color=c_s, lw=1.5, alpha=0.6,
                    label=f'S ({tag})')
            Gmax_ref = max(df3[np.isclose(df3['lb_tilde'], 0.4)]
                          .groupby('p_xl')['G_tilde'].median().max(), 1e-10)
            ax.plot(pxl, np.clip(agg['G_m'] / Gmax_ref, 0, None),
                    ls, color=c_g, lw=2, label=f'G ({tag})')
        ax.set_xlabel(r'$p_{xl}$')
        ax.set_ylabel(r'S, $\tilde{G}/\tilde{G}_{max}$')
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / 'fig4C_kappa0_control.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / 'fig4C_kappa0_control.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('  fig4C_kappa0_control')

        # G vs k for different lb
        fig, ax = plt.subplots(figsize=(7, 5.5))
        lb_vals = sorted(df3['lb_tilde'].unique())
        cmap_lb = plt.cm.plasma
        lb_norm = plt.Normalize(0, max(lb_vals))
        for lb in lb_vals:
            sub = df3[np.isclose(df3['lb_tilde'], lb)]
            agg = sub.groupby('p_xl').agg(k_m=('k_mean', 'mean'), G_m=('G_tilde', 'median'))
            Gmax = max(agg['G_m'].max(), 1e-10)
            col = cmap_lb(lb_norm(lb)) if lb > 0 else '#999'
            ax.plot(agg['k_m'], np.clip(agg['G_m'] / Gmax, 0, None),
                    '-o', ms=3, lw=2, color=col, label=f'lb={lb}')
        ax.set_xlabel(r'$\langle k \rangle$')
        ax.set_ylabel(r'$\tilde{G}/\tilde{G}_{max}$')
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / 'supp_bending_G_vs_k.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / 'supp_bending_G_vs_k.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('  supp_bending_G_vs_k')

    # --- Sweep 4: lambda consistency ---
    s4_path = BASE / 'sweep4_lambda_check' / 'all_results.json'
    if s4_path.exists():
        with open(s4_path) as f:
            s4 = json.load(f)
        rows = [{'lam_sim': r['params']['lam_sim'], 'p_xl': r['params']['p_xl'],
                 'S': r['topology']['S'], 'G_tilde': r['mechanics']['G_tilde']}
                for r in s4]
        df4 = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        for lam in sorted(df4['lam_sim'].unique()):
            sub = df4[np.isclose(df4['lam_sim'], lam)]
            agg = sub.groupby('p_xl').agg(G_m=('G_tilde', 'median'), S_m=('S', 'mean'))
            Gmax = max(agg['G_m'].max(), 1e-10)
            ax.plot(agg.index, agg['S_m'], '--', alpha=0.4, lw=1.5)
            ax.plot(agg.index, np.clip(agg['G_m'] / Gmax, 0, None),
                    '-o', ms=4, lw=2, label=f'lam={lam}')
        ax.set_xlabel(r'$p_{xl}$')
        ax.set_ylabel(r'S (dashed), $\tilde{G}/\tilde{G}_{max}$ (solid)')
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=10)
        ax.set_title('Dimensional consistency check')
        fig.tight_layout()
        fig.savefig(fig_dir / 'supp_lambda_check.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(fig_dir / 'supp_lambda_check.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('  supp_lambda_check')

    print(f'\nAll figures in {fig_dir}/')


def _interp_thresh(x, y, val):
    for i in range(len(x) - 1):
        if y[i] < val <= y[i + 1]:
            f = (val - y[i]) / (y[i + 1] - y[i])
            return x[i] + f * (x[i + 1] - x[i])
    return None


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep', type=int, nargs='+', default=None,
                   help='Which sweeps to run (1-4). Default: priority order.')
    p.add_argument('--figures-only', action='store_true')
    p.add_argument('--n-workers', type=int, default=N_WORKERS)
    p.add_argument('--quick', action='store_true',
                   help='Reduced seeds for testing (10/5)')
    args = p.parse_args()

    if args.figures_only:
        make_figures()
        return

    BASE.mkdir(parents=True, exist_ok=True)
    n1 = 10 if args.quick else 100
    n2 = 5 if args.quick else 50

    sweeps = args.sweep or [3, 1, 2, 4]  # priority: kappa0 first, then density

    for sw in sweeps:
        if sw == 1:
            if args.quick:
                sweep1_density(n_seeds=n1, a_values=[5, 8, 12], n_workers=args.n_workers)
            else:
                sweep1_density(n_seeds=n1, n_workers=args.n_workers)
        elif sw == 2:
            sweep2_size(n_seeds=n1, n_workers=args.n_workers)
        elif sw == 3:
            sweep3_bending(n_seeds=n2, n_workers=args.n_workers)
        elif sw == 4:
            sweep4_lambda_check(n_seeds=n2, n_workers=args.n_workers)

    make_figures()


if __name__ == '__main__':
    main()
