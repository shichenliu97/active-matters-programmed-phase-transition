#!/usr/bin/env python3
"""Generate individual high-res Figure 2 panels."""

import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'sandbox'))
from displacement_correlation import compute_displacement_correlations_along_x

OUT = Path(__file__).parent / 'sandbox' / 'figures' / '03182026' / 'figure_2' / 'panels'
OUT.mkdir(parents=True, exist_ok=True)

TRACKED = Path(__file__).parent / 'sandbox' / 'tracked_beads_v2'
PIV_1120 = Path(__file__).parent / 'sandbox' / 'figures' / '1120_piv'
PIXEL_SIZE = 0.43  # um/px
DT = 10  # s
ETA = 1e-3  # Pa·s
R_BEAD = 5e-6  # m
kBT = 4.1e-21  # J

plt.rcParams.update({
    'font.size': 22, 'axes.labelsize': 24, 'axes.titlesize': 24,
    'xtick.labelsize': 20, 'ytick.labelsize': 20,
    'axes.linewidth': 2.0, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'savefig.dpi': 400, 'figure.dpi': 150, 'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

time_cmap = plt.cm.viridis
t_norm = Normalize(vmin=5, vmax=95)

def load_pkl(p):
    with open(p, 'rb') as f: return pickle.load(f)

def add_colorbar(fig, ax, cmap, norm, label):
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(label, fontsize=22)
    return cb


# Work data (NearestVelocityPredict, top-20%, kBT) from work_summary.md
conditions = {
    5:  [1, 2, 3],
    30: [1, 2, 3],
    65: [1, 2, 3],
    85: [1, 2],
    95: [1, 2, 3],
}

# Per-position values from work_summary.md (NearestVelocityPredict)
work_data = {
    5:  [2445, 3276, 7951],
    30: [5173, 5942, 8446],
    65: [10298, 5296, 8148],
    85: [19665, 18755],
    95: [27468, 20610, 23076],
}

# ξ at onset from 1120 PIV
xi_data = {}  # time -> list of ξ values
for t, reps in conditions.items():
    xis = []
    for r in reps:
        pkl = PIV_1120 / f'{t}min_Pos{r-1}' / 'velocity_data.pkl'
        if not pkl.exists(): continue
        vd = load_pkl(pkl)
        try:
            _, _, _, _, _, xi = compute_displacement_correlations_along_x(
                vd, tau_frames=5, start_frame=0, end_frame=30)
            xis.append(xi)
        except:
            pass
    xi_data[t] = xis

print('Work data:')
for t, ws in work_data.items():
    print(f'  {t} min: {[f"{w:.0f}" for w in ws]} -> mean {np.mean(ws):.0f} kBT')
print('ξ data:')
for t, xs in xi_data.items():
    print(f'  {t} min: {[f"{x:.1f}" for x in xs]} -> mean {np.nanmean(xs):.1f} µm')


# Panel A: Work vs incubation time
print('\nPanel A...', flush=True)
fig, ax = plt.subplots(figsize=(7, 7))

times_sorted = sorted(work_data.keys())
for t in times_sorted:
    ws = work_data[t]
    col = time_cmap(t_norm(t))
    for w in ws:
        ax.scatter(t, w, c=[col], s=130, edgecolors='k', linewidth=0.5, zorder=2)

ax.set_xlabel('Assembly time (min)')
ax.set_ylabel(r'Work ($k_BT$)')
ax.set_xlim(0, 105)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_box_aspect(1)
add_colorbar(fig, ax, time_cmap, t_norm, 'Assembly time (min)')

plt.tight_layout()
fig.savefig(str(OUT / 'panel_A_v2.png'), dpi=400, bbox_inches='tight')
fig.savefig(str(OUT / 'panel_A_v2.pdf'), bbox_inches='tight')
plt.close()


# Panel B: Work vs ξ
print('Panel B...', flush=True)
fig, ax = plt.subplots(figsize=(7, 7))

for t in times_sorted:
    ws = work_data[t]
    xs = xi_data[t]
    if not ws or not xs:
        continue
    col = time_cmap(t_norm(t))
    mean_xi = np.nanmean(xs)
    for w in ws:
        ax.scatter(mean_xi, w, c=[col], s=130, edgecolors='k', linewidth=0.5, zorder=2)

ax.set_xlabel('ξ (µm)')
ax.set_ylabel(r'Work ($k_BT$)')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_box_aspect(1)
add_colorbar(fig, ax, time_cmap, t_norm, 'Assembly time (min)')

plt.tight_layout()
fig.savefig(str(OUT / 'panel_B_v2.png'), dpi=400, bbox_inches='tight')
fig.savefig(str(OUT / 'panel_B_v2.pdf'), bbox_inches='tight')
plt.close()


# Panel C: ξ evolution during recording
print('Panel C...', flush=True)
fig, ax = plt.subplots(figsize=(7, 7))

panel_c_conditions = {5: [0, 1, 2], 65: [0, 1, 2], 85: [0, 1]}
labels = {5: '5 min', 65: '65 min', 85: '85 min'}

max_start = 118 - 25
common_starts = list(range(0, max_start, 5))
common_tms = [(s + 10) * DT / 60 for s in common_starts]

for t, pos_list in panel_c_conditions.items():
    col = time_cmap(t_norm(t))
    all_curves = []
    for p in pos_list:
        pkl = PIV_1120 / f'{t}min_Pos{p}' / 'velocity_data.pkl'
        if not pkl.exists():
            continue
        vd = load_pkl(pkl)
        xis = []
        for s in common_starts:
            try:
                _, _, _, _, _, xi = compute_displacement_correlations_along_x(
                    vd, tau_frames=5, start_frame=s, end_frame=s + 20)
                xis.append(xi)
            except:
                xis.append(np.nan)
        all_curves.append(xis)

    if not all_curves:
        continue
    arr = np.array(all_curves)
    mean_xi = np.nanmean(arr, axis=0)
    std_xi = np.nanstd(arr, axis=0)
    ax.plot(common_tms, mean_xi, '-', color=col, lw=3.0, label=labels[t])
    ax.fill_between(common_tms, mean_xi - std_xi, mean_xi + std_xi, color=col, alpha=0.25)

ax.set_xlabel('Time into recording (min)')
ax.set_ylabel('ξ (µm)')
ax.set_ylim(0, 250)
ax.set_xlim(1, max(common_tms))
ax.legend(fontsize=18, frameon=False)
ax.set_box_aspect(1)

plt.tight_layout()
fig.savefig(str(OUT / 'panel_C_v2.png'), dpi=400, bbox_inches='tight')
fig.savefig(str(OUT / 'panel_C_v2.pdf'), bbox_inches='tight')
plt.close()


# Panel D: Representative bead trajectories
print('Panel D...', flush=True)

panel_d_conditions = [(5, '5 min'), (65, '65 min'), (85, '85 min')]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Find global spatial extent for consistent scale
all_ranges = []
for t, _ in panel_d_conditions:
    csv = TRACKED / f'1120_{t}min_1.csv'
    df = pd.read_csv(csv)
    df['x_um'] = df['x'] * PIXEL_SIZE
    df['y_um'] = df['y'] * PIXEL_SIZE
    all_ranges.append(df['x_um'].max() - df['x_um'].min())
    all_ranges.append(df['y_um'].max() - df['y_um'].min())
spatial_range = max(all_ranges) * 1.05

for ax, (t, label) in zip(axes, panel_d_conditions):
    csv = TRACKED / f'1120_{t}min_1.csv'
    df = pd.read_csv(csv)
    df['x_um'] = df['x'] * PIXEL_SIZE
    df['y_um'] = df['y'] * PIXEL_SIZE
    col = time_cmap(t_norm(t))

    for pid, grp in df.groupby('particle'):
        grp = grp.sort_values('frame')
        ax.plot(grp['x_um'].values, grp['y_um'].values, color=col, lw=1.2, alpha=0.5)

    cx = (df['x_um'].min() + df['x_um'].max()) / 2
    cy = (df['y_um'].min() + df['y_um'].max()) / 2
    half = spatial_range / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=22)
    ax.set_xlabel('x (µm)')

axes[0].set_ylabel('y (µm)')

plt.tight_layout()
fig.savefig(str(OUT / 'panel_D_v2.png'), dpi=400, bbox_inches='tight')
fig.savefig(str(OUT / 'panel_D_v2.pdf'), bbox_inches='tight')
plt.close()

print(f'Done — panels in {OUT}/')


# Composite: 2 rows x 3 cols
print('Composite figure...', flush=True)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: A, B, C
# Panel A
ax = axes[0, 0]
for t in times_sorted:
    ws = work_data[t]
    col = time_cmap(t_norm(t))
    for w in ws:
        ax.scatter(t, w, c=[col], s=130, edgecolors='k', linewidth=0.5, zorder=2)
ax.set_xlabel('Assembly time (min)')
ax.set_ylabel(r'Work ($k_BT$)')
ax.set_xlim(0, 105)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_box_aspect(1)

# Panel B
ax = axes[0, 1]
for t in times_sorted:
    ws = work_data[t]
    xs = xi_data[t]
    if not ws or not xs: continue
    col = time_cmap(t_norm(t))
    mean_xi = np.nanmean(xs)
    for w in ws:
        ax.scatter(mean_xi, w, c=[col], s=130, edgecolors='k', linewidth=0.5, zorder=2)
ax.set_xlabel('ξ (µm)')
ax.set_ylabel(r'Work ($k_BT$)')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_box_aspect(1)

# Panel C
ax = axes[0, 2]
for t, pos_list in panel_c_conditions.items():
    col = time_cmap(t_norm(t))
    all_curves = []
    for p in pos_list:
        pkl = PIV_1120 / f'{t}min_Pos{p}' / 'velocity_data.pkl'
        if not pkl.exists(): continue
        vd = load_pkl(pkl)
        xis = []
        for s in common_starts:
            try:
                _, _, _, _, _, xi = compute_displacement_correlations_along_x(
                    vd, tau_frames=5, start_frame=s, end_frame=s + 20)
                xis.append(xi)
            except:
                xis.append(np.nan)
        all_curves.append(xis)
    if not all_curves: continue
    arr = np.array(all_curves)
    mean_xi = np.nanmean(arr, axis=0)
    std_xi = np.nanstd(arr, axis=0)
    ax.plot(common_tms, mean_xi, '-', color=col, lw=3.0, label=labels[t])
    ax.fill_between(common_tms, mean_xi - std_xi, mean_xi + std_xi, color=col, alpha=0.25)
ax.set_xlabel('Time into recording (min)')
ax.set_ylabel('ξ (µm)')
ax.set_ylim(0, 250)
ax.set_xlim(1, max(common_tms))
ax.legend(fontsize=18, frameon=False)
ax.set_box_aspect(1)

# Row 2: D trajectories
all_ranges = []
for t, _ in panel_d_conditions:
    csv = TRACKED / f'1120_{t}min_1.csv'
    df = pd.read_csv(csv)
    df['x_um'] = df['x'] * PIXEL_SIZE
    df['y_um'] = df['y'] * PIXEL_SIZE
    all_ranges.append(df['x_um'].max() - df['x_um'].min())
    all_ranges.append(df['y_um'].max() - df['y_um'].min())
spatial_range = max(all_ranges) * 1.05

for idx, (t, label) in enumerate(panel_d_conditions):
    ax = axes[1, idx]
    csv = TRACKED / f'1120_{t}min_1.csv'
    df = pd.read_csv(csv)
    df['x_um'] = df['x'] * PIXEL_SIZE
    df['y_um'] = df['y'] * PIXEL_SIZE
    col = time_cmap(t_norm(t))
    for pid, grp in df.groupby('particle'):
        grp = grp.sort_values('frame')
        ax.plot(grp['x_um'].values, grp['y_um'].values, color=col, lw=1.2, alpha=0.5)
    cx = (df['x_um'].min() + df['x_um'].max()) / 2
    cy = (df['y_um'].min() + df['y_um'].max()) / 2
    half = spatial_range / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=22)
    ax.set_xlabel('x (µm)')
axes[1, 0].set_ylabel('y (µm)')

plt.tight_layout()
fig.savefig(str(OUT / 'figure2_composite.png'), dpi=400, bbox_inches='tight')
fig.savefig(str(OUT / 'figure2_composite.pdf'), bbox_inches='tight')
plt.close()
print('  Saved figure2_composite')
