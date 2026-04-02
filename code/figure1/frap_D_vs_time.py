"""FRAP diffusion coefficient D vs assembly time."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'simulation_figure_panels')
os.makedirs(OUT, exist_ok=True)

frap = pd.read_csv('/media/shichenliu/HHD_Storage_1013/Dropbox/Academics/PhD_phase/Thomson_Lab/Active_matter/local_to_global_pre-print/data/FRAP/frap_data/frap_data.csv')
frap_cols = [c for c in frap.columns if 'min' in c]

res = 0.294; r_circle = 120; w = res * r_circle

inc_times, D_vals = [], []
for col in frap_cols:
    t_inc = int(col.split('_')[0])
    vals = frap[col].dropna().values
    midval = (vals.max() - vals.min()) / 2
    idx_half = np.argmin(np.abs(vals - midval))
    tau_half = frap.loc[idx_half, 'Time (sec)']
    D = 0.88 * w**2 / (4 * tau_half)
    inc_times.append(t_inc)
    D_vals.append(D)

time_cmap = plt.cm.viridis
t_norm = Normalize(vmin=5, vmax=145)

plt.rcParams.update({
    'font.size': 18, 'axes.labelsize': 20, 'axes.titlesize': 20,
    'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(6, 4))
for ti, Di in zip(inc_times, D_vals):
    ax.scatter(ti, Di, c=[time_cmap(t_norm(ti))], s=110,
               edgecolors='k', linewidth=0.4, zorder=3)

ax.set_xlabel('Assembly time (min)')
ax.set_ylabel(r'D ($\mathrm{\mu m^2/s}$)')
ax.set_xlim(0, 120)
ax.set_ylim(0, None)

plt.tight_layout()
for ext in ('pdf', 'png'):
    path = os.path.join(OUT, f'frap_D_vs_time.{ext}')
    fig.savefig(path, dpi=400, bbox_inches='tight')
    print(f'Saved: {path}')
plt.close()
