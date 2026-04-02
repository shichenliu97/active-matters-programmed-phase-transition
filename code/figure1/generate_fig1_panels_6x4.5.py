#!/usr/bin/env python3
"""Generate individual high-res Figure 1 panels with colorbars."""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.optimize import fsolve

sys.path.insert(0, str(Path(__file__).parent / 'sandbox'))
from displacement_correlation import compute_displacement_correlations_along_x

OUT = Path(__file__).parent / 'sandbox' / 'figures' / '03182026' / 'figure_1' / 'panels_6x4.5'
OUT.mkdir(parents=True, exist_ok=True)

PIV_0104 = '/media/shichenliu/HHD_Storage_1013/Dropbox/Academics/PhD_phase/Thomson_Lab/Active_matter/local_to_global_pre-print/data/figure_1/0104PIV_data_0104'
MASTER = pd.read_csv(Path(__file__).parent / 'sandbox' / 'master_table.csv')
R_MAX = 430

plt.rcParams.update({
    'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 20,
    'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'savefig.dpi': 400, 'figure.dpi': 150, 'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Colormaps
time_cmap = plt.cm.viridis
prc1_cmap = plt.cm.Blues
times_0104 = [5, 10, 35, 65, 95, 120, 145]
t_norm = Normalize(vmin=5, vmax=145)
prc1_concs = [50, 75, 100, 150, 175, 200, 250]
p_norm = Normalize(vmin=50, vmax=250)

def load_pkl(p):
    with open(p, 'rb') as f: return pickle.load(f)

def add_colorbar(fig, ax, cmap, norm, label):
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(label, fontsize=16)
    return cb


# Panel B: Onset C(r) curves — time sweep
print('Panel B...', flush=True)
fig, ax = plt.subplots(figsize=(6, 4.5))

for t in times_0104:
    col = time_cmap(t_norm(t))
    curves, xis = [], []
    for rep in [1, 2, 3]:
        folder = f'5_min_{rep}_finer' if t == 5 else f'{t}_{rep}_finer'
        pkl = f'{PIV_0104}/{folder}/velocity_data.pkl'
        if not os.path.exists(pkl): continue
        vd = load_pkl(pkl)
        r, _, _, Dpn, _, xi = compute_displacement_correlations_along_x(
            vd, tau_frames=5, start_frame=0, end_frame=30)
        curves.append(Dpn); xis.append(xi)
        mask = r <= R_MAX
        ax.plot(r[mask], Dpn[mask], color=col, lw=0.8, alpha=0.3)
    if curves:
        mean_c = np.mean(curves, axis=0)
        ax.plot(r[mask], mean_c[mask], color=col, lw=2.2)
        mean_xi = np.nanmean(xis)
        if np.isfinite(mean_xi) and mean_xi <= R_MAX:
            ax.axvline(mean_xi, color=col, ls='--', lw=1.2, alpha=0.5)
            ax.plot(mean_xi, 1/np.e, 'o', color=col, ms=5, zorder=5)

ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
ax.axhline(1/np.e, color='gray', ls='--', lw=1.2, alpha=0.6)
ax.text(R_MAX - 5, 1/np.e + 0.03, '1/e', color='gray', fontsize=14, ha='right')
ax.set_xlim(0, R_MAX); ax.set_ylim(-0.2, 1.1)
ax.set_xlabel('Separation r (µm)'); ax.set_ylabel('C(r, τ)')

plt.tight_layout()
fig.savefig(str(OUT / 'panel_B.png'), dpi=400, bbox_inches='tight')
fig.savefig(str(OUT / 'panel_B.pdf'), bbox_inches='tight')
plt.close()


# Panel B2: ξ at onset vs incubation time
print('Panel B2...', flush=True)
fig, ax = plt.subplots(figsize=(6, 4.5))

onset_xis = {}
for t in times_0104:
    xis = []
    for rep in [1, 2, 3]:
        folder = f'5_min_{rep}_finer' if t == 5 else f'{t}_{rep}_finer'
        pkl = f'{PIV_0104}/{folder}/velocity_data.pkl'
        if not os.path.exists(pkl): continue
        vd = load_pkl(pkl)
        _, _, _, _, _, xi = compute_displacement_correlations_along_x(
            vd, tau_frames=5, start_frame=0, end_frame=30)
        xis.append(xi)
    onset_xis[t] = xis

for t, xis in onset_xis.items():
    col = time_cmap(t_norm(t))
    for xi in xis:
        ax.scatter(t, xi, c=[col], s=90, edgecolors='k', linewidth=0.3, zorder=2)

ax.set_xlabel('Assembly time (min)'); ax.set_ylabel('ξ (µm)')
ax.set_xlim(0, 155); ax.set_ylim(0, 200)

plt.tight_layout()
fig.savefig(str(OUT / 'panel_B2.png'), dpi=400, bbox_inches='tight')
fig.savefig(str(OUT / 'panel_B2.pdf'), bbox_inches='tight')
plt.close()


# Panel C: ξ evolution — 3 conditions, V1
print('Panel C...', flush=True)
fig, ax = plt.subplots(figsize=(6, 4.5))

max_start = 118 - 25
common_starts = list(range(0, max_start, 5))
common_tms = [(s + 10) * 10 / 60 for s in common_starts]

for t, label in [(5, '5 min'), (35, '35 min'), (120, '120 min')]:
    col = time_cmap(t_norm(t))
    all_curves = []
    for rep in [1, 2, 3]:
        folder = f'5_min_{rep}_finer' if t == 5 else f'{t}_{rep}_finer'
        pkl = f'{PIV_0104}/{folder}/velocity_data.pkl'
        if not os.path.exists(pkl): continue
        vd = load_pkl(pkl)
        xis = []
        for s in common_starts:
            try:
                _, _, _, _, _, xi = compute_displacement_correlations_along_x(vd, tau_frames=5, start_frame=s, end_frame=s + 20)
                xis.append(xi)
            except:
                xis.append(np.nan)
        all_curves.append(xis)
    arr = np.array(all_curves)
    mean_xi = np.nanmean(arr, axis=0)
    std_xi = np.nanstd(arr, axis=0)
    ax.plot(common_tms, mean_xi, '-', color=col, lw=2.5, label=label)
    ax.fill_between(common_tms, mean_xi - std_xi, mean_xi + std_xi, color=col, alpha=0.25)

ax.axhline(35, color='gray', ls=':', lw=1, alpha=0.4)
ax.axhline(120, color='gray', ls=':', lw=1, alpha=0.4)
ax.set_xlabel('Time into recording (min)'); ax.set_ylabel('ξ (µm)')
ax.set_ylim(0, 200); ax.set_xlim(1, max(common_tms))
ax.legend(fontsize=12, frameon=False)

plt.tight_layout()
fig.savefig(str(OUT / 'panel_C.png'), dpi=400, bbox_inches='tight')
fig.savefig(str(OUT / 'panel_C.pdf'), bbox_inches='tight')
plt.close()


# Panel D: FRAP
print('Panel D...', flush=True)
fig, ax = plt.subplots(figsize=(6, 4.5))

frap = pd.read_csv('/media/shichenliu/HHD_Storage_1013/Dropbox/Academics/PhD_phase/Thomson_Lab/Active_matter/local_to_global_pre-print/data/FRAP/frap_data/frap_data.csv')
frap_cols = [c for c in frap.columns if 'min' in c]
res = 0.294; r_circle = 120; w = res * r_circle
k_B = 1.38e-23; T = 25 + 273.15; eta = 2e-3; r_fil = 12.5e-9

def calc_length(D_um2s):
    D_m2s = D_um2s * 1e-12
    def eq(logL):
        L = np.exp(logL)
        return (2 * np.pi * eta * L * D_m2s) / (k_B * T) - logL + np.log(2 * r_fil) + 0.2
    logL = fsolve(eq, -12)[0]
    return np.exp(logL) * 1e6

inc_times, lengths = [], []
for col in frap_cols:
    t_inc = int(col.split('_')[0])
    vals = frap[col].dropna().values
    midval = (vals.max() - vals.min()) / 2
    idx_half = np.argmin(np.abs(vals - midval))
    tau_half = frap.loc[idx_half, 'Time (sec)']
    D = 0.88 * w ** 2 / (4 * tau_half)
    L = calc_length(D)
    inc_times.append(t_inc); lengths.append(L)

t_arr = np.array(inc_times, dtype=float); L_arr = np.array(lengths)
slope, intercept = np.polyfit(t_arr, L_arr, 1)

for ti, Li in zip(inc_times, lengths):
    ax.scatter(ti, Li, c=[time_cmap(t_norm(ti))], s=110, edgecolors='k', linewidth=0.4, zorder=3)
t_fit = np.linspace(0, 120, 100)
ax.plot(t_fit, slope * t_fit + intercept, 'k--', lw=1.5, alpha=0.6,
        label=f'{slope * 60:.2f} µm/hr')
ax.set_xlabel('Assembly time (min)'); ax.set_ylabel('Bundle length (µm)')
ax.set_xlim(0, 120); ax.set_ylim(0, 6)
ax.legend(fontsize=12, frameon=False)

plt.tight_layout()
fig.savefig(str(OUT / 'panel_D.png'), dpi=400, bbox_inches='tight')
fig.savefig(str(OUT / 'panel_D.pdf'), bbox_inches='tight')
plt.close()


# Panel E: ξ vs PRC1 — plasma colormap
print('Panel E...', flush=True)
fig, ax = plt.subplots(figsize=(6, 4.5))

pdf = MASTER[(MASTER['dataset'] == 'prc1') & (MASTER['PRC1_nM'] <= 250)]
np.random.seed(42)
for _, row in pdf.iterrows():
    jit = np.random.uniform(-3, 3)
    col = prc1_cmap(p_norm(row['PRC1_nM']))
    ax.scatter(row['PRC1_nM'] + jit, row['xi_tau50s'], c=[col], s=90,
               edgecolors='k', linewidth=0.3, zorder=2)
ax.set(xlabel='[PRC1] (nM)', ylabel='ξ (µm)', xlim=(25, 275), ylim=(0, 170))
ax.grid(alpha=0.15)
add_colorbar(fig, ax, prc1_cmap, p_norm, '[PRC1] (nM)')

plt.tight_layout()
fig.savefig(str(OUT / 'panel_E.png'), dpi=400, bbox_inches='tight')
fig.savefig(str(OUT / 'panel_E.pdf'), bbox_inches='tight')
plt.close()

print(f'Done — panels in {OUT}/')
