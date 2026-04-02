#!/usr/bin/env python3
"""Extended p_xl + lambda sweep for k-collapse analysis."""
import numpy as np
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import label, binary_dilation
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

LAMBDAS = [2.0, 2.5, 3.5]
PXL_PRIMARY = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
               0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]
N_SEEDS = 100

PXL_CONTROL = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
N_SEEDS_CTRL = 50
LAM_CTRL = 3.5

# Lambda sweep for collapse
LAM_SWEEP = np.linspace(0.5, 5.0, 30).tolist()
PXL_SWEEP = 0.5
N_SEEDS_LAM = 100

N_WORKERS = 128


def compute_voids(net, grid_size=200):
    """Rasterize edges, find void areas in complement."""
    if net.n_edges < 3 or net.n_nodes < 3:
        return []
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    scale = grid_size / net.L
    for ei in range(net.n_edges):
        n1, n2 = net.edges[ei]
        p1 = net.pos[n1]
        d = net.pos[n2] - p1
        d -= net.L * np.round(d / net.L)
        n_pts = max(int(np.linalg.norm(d) * scale * 2), 3)
        ts = np.linspace(0, 1, n_pts)
        pts = p1[None, :] + ts[:, None] * d[None, :]
        px = (pts[:, 0] % net.L * scale).astype(int) % grid_size
        py = (pts[:, 1] % net.L * scale).astype(int) % grid_size
        grid[py, px] = True
    grid = binary_dilation(grid)
    labeled, n_lab = label(~grid)
    px_area = (net.L / grid_size) ** 2
    min_a = 4 * px_area
    counts = np.bincount(labeled.ravel())
    return sorted([float(c * px_area) for c in counts[1:] if c * px_area > min_a],
                  reverse=True)


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

    k_per = net.crosslinks_per_fiber()
    k_mean = float(k_per.mean())

    n_f = net.n_f
    n_active = len(net._fiber_xl)
    f_isolated = 1.0 - n_active / n_f

    G = net.shear_modulus(gamma=gamma)

    voids = compute_voids(net)

    return {
        'lambda': float(lam), 'p_xl': float(p_xl), 'seed': int(seed),
        'kappa': float(kappa),
        'S': float(S), 'z': float(z), 'G': float(G),
        'N_nodes': int(n_nodes), 'N_edges': int(n_edges),
        'N_components': int(n_comp), 'beta1': int(beta1),
        'k_mean': float(k_mean),
        'n_fibers': int(n_f), 'f_isolated': float(f_isolated),
        'n_voids': len(voids),
        'mean_void_area': float(np.mean(voids)) if voids else 0.0,
        'void_areas': voids,
    }


def run_sweep(jobs, desc, n_workers=N_WORKERS):
    cs = max(1, len(jobs) // (n_workers * 4))
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for r in tqdm(pool.map(run_one, jobs, chunksize=cs),
                      total=len(jobs), desc=desc):
            results.append(r)
    return results


def run_all(n_workers):
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. p_xl sweep at 3 lambdas
    n = len(LAMBDAS) * len(PXL_PRIMARY) * N_SEEDS
    print()
    print(f'P_XL SWEEP: {len(LAMBDAS)} lam x {len(PXL_PRIMARY)} p_xl x {N_SEEDS} seeds = {n}')
    print()
    jobs = [(lam, p, s, KAPPA, K_SHAPE, BUDGET, L_BOX, GAMMA)
            for lam in LAMBDAS for p in PXL_PRIMARY for s in range(N_SEEDS)]
    pxl_results = run_sweep(jobs, 'p_xl sweep', n_workers)
    with open(OUT / 'phase2_pxl.json', 'w') as f:
        json.dump(pxl_results, f)
    print(f'  Saved {len(pxl_results)} results')

    # 2. kappa=0 control
    n2 = len(PXL_CONTROL) * N_SEEDS_CTRL
    print()
    print(f'CONTROL (kappa=0): {n2} runs')
    print()
    jobs = [(LAM_CTRL, p, s, 0.0, K_SHAPE, BUDGET, L_BOX, GAMMA)
            for p in PXL_CONTROL for s in range(N_SEEDS_CTRL)]
    ctrl = run_sweep(jobs, 'Control kappa=0', n_workers)
    with open(OUT / 'phase2_control.json', 'w') as f:
        json.dump(ctrl, f)

    # 3. Lambda sweep for collapse
    n3 = len(LAM_SWEEP) * N_SEEDS_LAM
    print()
    print(f'LAMBDA SWEEP: {len(LAM_SWEEP)} lam x {N_SEEDS_LAM} seeds = {n3}')
    print()
    jobs = [(lam, PXL_SWEEP, s, KAPPA, K_SHAPE, BUDGET, L_BOX, GAMMA)
            for lam in LAM_SWEEP for s in range(N_SEEDS_LAM)]
    lam_results = run_sweep(jobs, 'Lambda sweep', n_workers)
    with open(OUT / 'phase2_lambda.json', 'w') as f:
        json.dump(lam_results, f)
    print(f'  Saved {len(lam_results)} results')

    return pxl_results, ctrl, lam_results


def _find_thresh(x, y, val):
    for i in range(len(x) - 1):
        if y[i] < val <= y[i + 1]:
            f = (val - y[i]) / (y[i + 1] - y[i])
            return x[i] + f * (x[i + 1] - x[i])
    return None


def make_figures(pxl_data=None, ctrl_data=None, lam_data=None):
    import pandas as pd
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if pxl_data is None:
        with open(OUT / 'phase2_pxl.json') as f:
            pxl_data = json.load(f)
    if ctrl_data is None:
        with open(OUT / 'phase2_control.json') as f:
            ctrl_data = json.load(f)
    if lam_data is None:
        with open(OUT / 'phase2_lambda.json') as f:
            lam_data = json.load(f)

    df = pd.DataFrame(pxl_data)
    df_ctrl = pd.DataFrame(ctrl_data)
    df_lam = pd.DataFrame(lam_data)

    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'axes.linewidth': 1.5, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'savefig.dpi': 400, 'figure.dpi': 150, 'font.family': 'sans-serif',
        'axes.spines.top': False,
    })
    c_S, c_G, c_k = '#2166AC', '#B2182B', '#E6550D'
    lam_colors = {2.0: '#2ca02c', 2.5: c_k, 3.5: c_S}
    lam_markers = {2.0: 'D', 2.5: 'o', 3.5: 's'}

    # Print detailed table
    print(f'\n{"="*90}')
    print(f'{"lam":>5} {"p_xl":>5} {"S":>7} {"G_med":>10} {"<k>":>6} {"z":>5} '
          f'{"f_iso":>6} {"voids":>5} {"N":>5} {"b1":>5}')
    print()
    for lam in sorted(df['lambda'].unique()):
        for pxl in sorted(df[df['lambda'] == lam]['p_xl'].unique()):
            s = df[(df['lambda'] == lam) & (df['p_xl'] == pxl)]
            print(f'{lam:5.1f} {pxl:5.2f} {s["S"].mean():7.3f} '
                  f'{s["G"].median():10.4f} {s["k_mean"].mean():6.1f} '
                  f'{s["z"].mean():5.2f} {s["f_isolated"].mean():6.2f} '
                  f'{s["n_voids"].mean():5.0f} {s["N_nodes"].mean():5.0f} '
                  f'{s["beta1"].mean():5.0f}')
    print()

    #Fig 1: S and G/G_max vs p_xl (3 lambdas)    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, lam in zip(axes, LAMBDAS):
        sub = df[df['lambda'] == lam]
        agg = sub.groupby('p_xl').agg(
            S_m=('S', 'mean'), S_s=('S', 'std'),
            G_med=('G', 'median'),
            G_q25=('G', lambda x: np.percentile(x, 25)),
            G_q75=('G', lambda x: np.percentile(x, 75)))
        pxl = agg.index.values
        S_m, S_s = agg['S_m'].values, agg['S_s'].values
        G_m = agg['G_med'].values
        Gmax = max(G_m.max(), 1e-10)
        Gn = G_m / Gmax
        Gn_lo, Gn_hi = agg['G_q25'].values / Gmax, agg['G_q75'].values / Gmax

        pc = _find_thresh(pxl, S_m, 0.5)
        pr = _find_thresh(pxl, Gn, 0.1)
        if pc and pr and pr > pc:
            ax.axvspan(pc, pr, alpha=0.10, color='gold', zorder=0,
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

        ax.set_xlabel(r'$p_{\mathrm{xl}}$')
        ax.set_ylabel('S', color=c_S)
        ax.tick_params(axis='y', labelcolor=c_S)
        ax.set_ylim(-0.05, 1.15)
        gap_str = f', gap={pr - pc:.2f}' if pc and pr and pr > pc else ''
        ax.set_title(r'$\lambda$ = ' + f'{lam}{gap_str}')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_pxl_SG.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig1_pxl_SG.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig1_pxl_SG')

    #Fig 2: <k> vs p_xl with Maxwell line    fig, ax = plt.subplots(figsize=(7, 5.5))
    for lam in LAMBDAS:
        sub = df[df['lambda'] == lam]
        agg = sub.groupby('p_xl').agg(
            k_m=('k_mean', 'mean'), k_s=('k_mean', 'std'),
            G_med=('G', 'median'))
        pxl = agg.index.values
        k_m, k_s = agg['k_m'].values, agg['k_s'].values
        G_m = agg['G_med'].values

        clr = lam_colors[lam]
        mkr = lam_markers[lam]
        ax.fill_between(pxl, k_m - k_s, k_m + k_s, alpha=0.12, color=clr)
        ax.plot(pxl, k_m, f'-{mkr}', ms=5, lw=2, color=clr,
                label=r'$\lambda$=' + f'{lam}')

        Gmax = max(G_m.max(), 1e-10)
        pr = _find_thresh(pxl, G_m / Gmax, 0.1)
        if pr:
            idx = np.argmin(np.abs(pxl - pr))
            ax.plot(pxl[idx], k_m[idx], '*', color=c_G, ms=15, zorder=5)

    ax.axhline(3.0, color='k', ls='--', alpha=0.5, lw=1.5,
               label=r'$\langle k \rangle$ = 3 (Maxwell)')
    ax.set_xlabel(r'$p_{\mathrm{xl}}$')
    ax.set_ylabel(r'$\langle k \rangle$')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_k_mean.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig2_k_mean.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig2_k_mean')

    #Fig 3: kappa=0 control    fig, ax = plt.subplots(figsize=(7, 5.5))
    sub_k1 = df[df['lambda'] == LAM_CTRL]
    agg_k1 = sub_k1.groupby('p_xl').agg(S_m=('S', 'mean'), G_med=('G', 'median'))
    pxl_k1 = agg_k1.index.values
    Gmax_k1 = max(agg_k1['G_med'].values.max(), 1e-10)

    ax.plot(pxl_k1, agg_k1['S_m'].values, '--', color=c_S, lw=1.5, alpha=0.5,
            label=r'S ($\kappa$=1)')
    ax.plot(pxl_k1, np.clip(agg_k1['G_med'].values / Gmax_k1, 0, None),
            '-s', color=c_G, ms=4, lw=2, label=r'G/G$_{\mathrm{max}}$ ($\kappa$=1)')

    agg_k0 = df_ctrl.groupby('p_xl').agg(S_m=('S', 'mean'), G_med=('G', 'median'))
    pxl_k0 = agg_k0.index.values
    ax.plot(pxl_k0, agg_k0['S_m'].values, '--', color='#999', lw=1.5, alpha=0.5,
            label=r'S ($\kappa$=0)')
    ax.plot(pxl_k0, np.clip(agg_k0['G_med'].values / Gmax_k1, 0, None),
            '-o', color='#999', ms=4, lw=2, label=r'G/G$_{\mathrm{max}}$ ($\kappa$=0)')
    ax.set_xlabel(r'$p_{\mathrm{xl}}$')
    ax.set_ylabel(r'S,  G/G$_{\mathrm{max}}$')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=10)
    ax.set_title(r'Bending ($\kappa > 0$) required for rigidity')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_kappa0.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig3_kappa0.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig3_kappa0')

    #Fig 4: ⟨k⟩ COLLAPSE — the key new figure    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: S vs <k>
    ax = axes[0]
    # p_xl sweep data (3 lambdas)
    for lam in LAMBDAS:
        sub = df[df['lambda'] == lam]
        agg = sub.groupby('p_xl').agg(k_m=('k_mean', 'mean'), S_m=('S', 'mean'))
        clr = lam_colors[lam]
        mkr = lam_markers[lam]
        ax.plot(agg['k_m'].values, agg['S_m'].values, f'-{mkr}', ms=5, lw=2,
                color=clr, alpha=0.8,
                label=rf'$p_{{xl}}$ sweep, $\lambda$={lam}')

    # Lambda sweep data
    agg_l = df_lam.groupby('lambda').agg(k_m=('k_mean', 'mean'), S_m=('S', 'mean'))
    ax.plot(agg_l['k_m'].values, agg_l['S_m'].values, 'k-^', ms=5, lw=2,
            alpha=0.8, label=rf'$\lambda$ sweep, $p_{{xl}}$={PXL_SWEEP}')

    ax.axvline(1.0, color='gray', ls=':', alpha=0.4)
    ax.set_xlabel(r'$\langle k \rangle$')
    ax.set_ylabel('S')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title(r'Connectivity vs $\langle k \rangle$')

    # Panel B: G/G_max vs <k>
    ax = axes[1]
    for lam in LAMBDAS:
        sub = df[df['lambda'] == lam]
        agg = sub.groupby('p_xl').agg(k_m=('k_mean', 'mean'), G_med=('G', 'median'))
        G_m = agg['G_med'].values
        Gmax = max(G_m.max(), 1e-10)
        clr = lam_colors[lam]
        mkr = lam_markers[lam]
        ax.plot(agg['k_m'].values, np.clip(G_m / Gmax, 0, None),
                f'-{mkr}', ms=5, lw=2, color=clr, alpha=0.8,
                label=rf'$p_{{xl}}$ sweep, $\lambda$={lam}')

    agg_l = df_lam.groupby('lambda').agg(k_m=('k_mean', 'mean'), G_med=('G', 'median'))
    G_lam = agg_l['G_med'].values
    Gmax_l = max(G_lam.max(), 1e-10)
    ax.plot(agg_l['k_m'].values, np.clip(G_lam / Gmax_l, 0, None),
            'k-^', ms=5, lw=2, alpha=0.8,
            label=rf'$\lambda$ sweep, $p_{{xl}}$={PXL_SWEEP}')

    ax.axvline(3.0, color='k', ls='--', alpha=0.5, lw=1.5,
               label=r'$\langle k \rangle$ = 3')
    ax.set_xlabel(r'$\langle k \rangle$')
    ax.set_ylabel(r'G / G$_{\mathrm{max}}$')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title(r'Rigidity vs $\langle k \rangle$')

    fig.suptitle(r'$\langle k \rangle$ collapse: both sweep directions on same curve',
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_k_collapse.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig4_k_collapse.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig4_k_collapse')

    #Fig 5: f_isolated vs p_xl    fig, ax = plt.subplots(figsize=(7, 5.5))
    for lam in LAMBDAS:
        sub = df[df['lambda'] == lam]
        agg = sub.groupby('p_xl').agg(
            fi_m=('f_isolated', 'mean'), fi_s=('f_isolated', 'std'))
        pxl = agg.index.values
        fi_m, fi_s = agg['fi_m'].values, agg['fi_s'].values
        clr = lam_colors[lam]
        mkr = lam_markers[lam]
        ax.fill_between(pxl, fi_m - fi_s, fi_m + fi_s, alpha=0.12, color=clr)
        ax.plot(pxl, fi_m, f'-{mkr}', ms=5, lw=2, color=clr,
                label=r'$\lambda$=' + f'{lam}')
    ax.set_xlabel(r'$p_{\mathrm{xl}}$')
    ax.set_ylabel('Fraction of isolated fibers')
    ax.legend(fontsize=10)
    ax.set_title('Fibers with zero crosslinks')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_f_isolated.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig5_f_isolated.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig5_f_isolated')

    #Fig 6: Void area distributions    pxl_show = [0.15, 0.30, 0.50, 0.70, 1.00]
    lam_void = 3.5  # use densest lambda for best void statistics
    fig, axes = plt.subplots(1, len(pxl_show), figsize=(4 * len(pxl_show), 4),
                             sharey=True)
    cmap = plt.cm.viridis
    for i, pxl_val in enumerate(pxl_show):
        ax = axes[i]
        sub = df[(df['lambda'] == lam_void) &
                 (np.abs(df['p_xl'] - pxl_val) < 0.01)]
        all_areas = []
        for _, row in sub.iterrows():
            all_areas.extend(row['void_areas'])
        if all_areas:
            all_areas = np.array(all_areas)
            # plot sqrt(area) ~ void diameter
            diams = np.sqrt(all_areas)
            ax.hist(diams, bins=25, color=cmap(i / len(pxl_show)),
                    alpha=0.7, edgecolor='white', lw=0.5)
            ax.set_xlabel(r'$\sqrt{A_{\mathrm{void}}}$')
            md = np.median(diams)
            ax.axvline(md, color='k', ls=':', alpha=0.5)
            ax.text(0.95, 0.95,
                    f'n={len(all_areas)}\nmed={md:.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9)
        ax.set_title(f'$p_{{xl}}$={pxl_val:.2f}')
    axes[0].set_ylabel('Count')
    fig.suptitle(rf'Void size distributions ($\lambda$={lam_void})',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_void_distributions.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig6_void_distributions.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig6_void_distributions')

    #Fig 7: Sim vs experiment    exp_time = np.array([5, 60, 95, 115, 125])
    exp_xi = np.array([35, 100, 110, 112, 115])
    exp_work = np.array([4600, 6500, 7900, 19200, 23700])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    sub = df[df['lambda'] == 2.5]
    agg = sub.groupby('p_xl').agg(
        S_m=('S', 'mean'), S_s=('S', 'std'),
        G_med=('G', 'median'),
        G_q25=('G', lambda x: np.percentile(x, 25)),
        G_q75=('G', lambda x: np.percentile(x, 75)))
    pxl = agg.index.values
    S_m, S_s = agg['S_m'].values, agg['S_s'].values
    G_m = agg['G_med'].values
    Gmax = max(G_m.max(), 1e-10)
    Gn = G_m / Gmax

    pc = _find_thresh(pxl, S_m, 0.5)
    pr = _find_thresh(pxl, Gn, 0.1)
    if pc and pr and pr > pc:
        ax.axvspan(pc, pr, alpha=0.10, color='gold', zorder=0)

    ax.fill_between(pxl, np.clip(S_m - S_s, 0, 1),
                    np.clip(S_m + S_s, 0, 1), alpha=0.15, color=c_S)
    ax.plot(pxl, S_m, '-o', color=c_S, ms=5, lw=2, label='S')
    ax2 = ax.twinx()
    ax2.fill_between(pxl, np.clip(agg['G_q25'].values / Gmax, 0, 1.3),
                     np.clip(agg['G_q75'].values / Gmax, 0, 1.3),
                     alpha=0.15, color=c_G)
    ax2.plot(pxl, np.clip(Gn, 0, None), '-s', color=c_G, ms=5, lw=2,
             label=r'G/G$_{\mathrm{max}}$')
    ax.set_xlabel(r'$p_{\mathrm{xl}}$')
    ax.set_ylabel('S', color=c_S)
    ax2.set_ylabel(r'G/G$_{\mathrm{max}}$', color=c_G)
    ax.set_ylim(-0.05, 1.15); ax2.set_ylim(-0.05, 1.15)
    ax2.spines['top'].set_visible(False)
    ax.set_title(r'A. Simulation ($\lambda$=2.5)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    ax = axes[1]
    ax.plot(exp_time, exp_xi / exp_xi.max(), '-o', color=c_S, ms=8, lw=2,
            label=r'$\xi/\xi_{\mathrm{max}}$')
    ax2 = ax.twinx()
    ax2.plot(exp_time, exp_work / exp_work.max(), '-s', color=c_G, ms=8, lw=2,
             label=r'$W/W_{\mathrm{max}}$')
    ax.axvspan(35, 95, alpha=0.10, color='gold', zorder=0)
    ax.set_xlabel('Incubation time (min)')
    ax.set_ylabel(r'$\xi/\xi_{\mathrm{max}}$', color=c_S)
    ax2.set_ylabel(r'$W/W_{\mathrm{max}}$', color=c_G)
    ax.set_ylim(-0.05, 1.15); ax2.set_ylim(-0.05, 1.15)
    ax2.spines['top'].set_visible(False)
    ax.set_title('B. Experiment')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig7_sim_vs_exp.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig7_sim_vs_exp.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved fig7_sim_vs_exp')

    #Summary    print()
    print('SUMMARY')
    print()
    for lam in LAMBDAS:
        sub = df[df['lambda'] == lam]
        agg = sub.groupby('p_xl').agg(
            S_m=('S', 'mean'), G_med=('G', 'median'),
            k_m=('k_mean', 'mean'), fi_m=('f_isolated', 'mean'))
        pxl = agg.index.values
        S, G, k = agg['S_m'].values, agg['G_med'].values, agg['k_m'].values
        Gmax = max(G.max(), 1e-10)
        Gn = G / Gmax
        pc = _find_thresh(pxl, S, 0.5)
        pr = _find_thresh(pxl, Gn, 0.1)
        gap = (pr - pc) if pc and pr else None
        print(f'\n  lambda={lam}:')
        print(f'    p_xl_conn  (S>0.5) = {pc:.3f}' if pc else '    p_xl_conn: not found')
        print(f'    p_xl_rigid (G>0.1) = {pr:.3f}' if pr else '    p_xl_rigid: not found')
        print(f'    gap = {gap:.3f}' if gap else '    gap: N/A')
        if pr:
            idx = np.argmin(np.abs(pxl - pr))
            print(f'    <k> at rigidity = {k[idx]:.2f}')
            print(f'    f_isolated at rigidity = {agg["fi_m"].values[idx]:.2f}')

    # Lambda sweep thresholds
    agg_l = df_lam.groupby('lambda').agg(
        S_m=('S', 'mean'), G_med=('G', 'median'), k_m=('k_mean', 'mean'))
    lam_arr = agg_l.index.values
    S_l, G_l, k_l = agg_l['S_m'].values, agg_l['G_med'].values, agg_l['k_m'].values
    Gmax_l = max(G_l.max(), 1e-10)
    lc = _find_thresh(lam_arr, S_l, 0.5)
    lr = _find_thresh(lam_arr, G_l / Gmax_l, 0.1)
    print(f'\n  Lambda sweep (p_xl={PXL_SWEEP}):')
    print(f'    lam_conn = {lc:.2f}' if lc else '    lam_conn: not found')
    print(f'    lam_rigid = {lr:.2f}' if lr else '    lam_rigid: not found')
    if lc:
        idx_c = np.argmin(np.abs(lam_arr - lc))
        print(f'    <k> at connectivity = {k_l[idx_c]:.2f}')
    if lr:
        idx_r = np.argmin(np.abs(lam_arr - lr))
        print(f'    <k> at rigidity = {k_l[idx_r]:.2f}')

    print(f'\n  kappa=0 control: max G = {df_ctrl["G"].max():.6f}')
    print()
    print(f'\nAll figures: {FIG_DIR}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--figures-only', action='store_true')
    p.add_argument('--n-workers', type=int, default=N_WORKERS)
    args = p.parse_args()

    if args.figures_only:
        make_figures()
        return

    pxl, ctrl, lam = run_all(args.n_workers)
    make_figures(pxl, ctrl, lam)


if __name__ == '__main__':
    main()
