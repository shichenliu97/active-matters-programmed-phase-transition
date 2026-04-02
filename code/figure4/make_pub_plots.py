"""Generate publication-quality plots from sweep results."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'publication_plots')
os.makedirs(OUT, exist_ok=True)

def load(tag):
    with open(os.path.join(BASE, f'sweep_{tag}.json')) as f:
        return json.load(f)

def agg(data):
    df = pd.DataFrame(data)
    a = df.groupby('lam').agg(['mean','std'])
    return (a.index.values, a[('S','mean')].values, a[('S','std')].values,
            a[('G','mean')].values, a[('G','std')].values,
            a[('z','mean')].values, a[('z','std')].values,
            a[('density','mean')].values)

#1: Main result B=300lam, S_m, S_s, G_m, G_s, z_m, z_s, d_m = agg(load('B300.0_k1.0_pxl0.5'))
Gx = max(G_m.max(), 1e-10); Gn = G_m/Gx; Gn_s = G_s/Gx

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
ax.fill_between(lam, np.clip(S_m-S_s,0,1), np.clip(S_m+S_s,0,1), alpha=0.15, color='#1f77b4')
ax.plot(lam, S_m, '-o', color='#1f77b4', ms=4, lw=2, label='S (connectivity)')
ax.fill_between(lam, np.clip(Gn-Gn_s,0,1.2), np.clip(Gn+Gn_s,0,1.2), alpha=0.15, color='#d62728')
ax.plot(lam, np.clip(Gn,0,None), '-s', color='#d62728', ms=4, lw=2, label=r'G/G$_{max}$ (rigidity)')
ax.axvspan(1.04, 2.70, alpha=0.08, color='gold', label='gap')
ax.set_ylim(-0.05, 1.15); ax.set_xlabel(r'$\lambda$', fontsize=12)
ax.set_ylabel(r'S,  G/G$_{max}$', fontsize=12); ax.legend(fontsize=9); ax.set_title('Two sequential transitions')

ax = axes[1]
ax.fill_between(lam, z_m-z_s, z_m+z_s, alpha=0.15, color='k')
ax.plot(lam, z_m, 'k-o', ms=4, lw=2)
ax.axhline(4, color='red', ls='--', alpha=0.5, label='z=4 (Maxwell)')
ax.set_xlabel(r'$\lambda$', fontsize=12); ax.set_ylabel('2E/N', fontsize=12)
ax.legend(fontsize=9); ax.set_title('Coordination number')

ax = axes[2]
ax.plot(lam, d_m, '-o', color='#2ca02c', ms=4, lw=2)
ax.set_xlabel(r'$\lambda$', fontsize=12); ax.set_ylabel(r'nodes / L$^2$', fontsize=12)
ax.set_title('Intersection density')

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_main_B300.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_main_B300.png'), dpi=200, bbox_inches='tight')
plt.close(); print('saved fig_main_B300')

#2: p_xl sensitivityfig, ax = plt.subplots(figsize=(7, 5))
colors = {'0.3': '#e377c2', '0.5': '#d62728', '0.7': '#ff7f0e', '1.0': '#2ca02c'}
for pxl in ['0.3', '0.5', '0.7', '1.0']:
    lam, S_m, _, G_m, _, _, _, _ = agg(load(f'B300.0_k1.0_pxl{pxl}'))
    Gx = max(G_m.max(), 1e-10)
    ax.plot(lam, S_m, '--', color=colors[pxl], alpha=0.4, lw=1.5)
    ax.plot(lam, np.clip(G_m/Gx, 0, None), '-s', color=colors[pxl], ms=3, lw=2,
            label=r'p$_{xl}$=' + pxl)
ax.set_ylim(-0.05, 1.15); ax.set_xlabel(r'$\lambda$', fontsize=12)
ax.set_ylabel(r'S (dashed),  G/G$_{max}$ (solid)', fontsize=12)
ax.legend(fontsize=10, title='Crosslink fraction')
ax.set_title(r'Gap width depends on $p_{xl}$')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_pxl_sensitivity.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_pxl_sensitivity.png'), dpi=200, bbox_inches='tight')
plt.close(); print('saved fig_pxl_sensitivity')

#3: κ sensitivityfig, axes = plt.subplots(1, 2, figsize=(12, 5))
kappas = [('0.0', '#999999'), ('0.1', '#9467bd'), ('1.0', '#d62728'), ('10.0', '#1f77b4')]

ax = axes[0]
for kstr, c in kappas:
    lam, _, _, G_m, _, _, _, _ = agg(load(f'B300.0_k{kstr}_pxl0.5'))
    Gx = max(G_m.max(), 1e-10)
    ax.plot(lam, np.clip(G_m/Gx, 0, None), '-o', color=c, ms=3, lw=2, label=r'$\kappa$='+kstr)
ax.set_ylim(-0.05, 1.15); ax.set_xlabel(r'$\lambda$', fontsize=12)
ax.set_ylabel(r'G/G$_{max}$', fontsize=12); ax.legend(fontsize=10)
ax.set_title(r'Rigidity onset independent of $\kappa$')

ax = axes[1]
for kstr, c in kappas:
    lam, _, _, G_m, _, _, _, _ = agg(load(f'B300.0_k{kstr}_pxl0.5'))
    ax.plot(lam, np.maximum(G_m, 1e-4), '-o', color=c, ms=3, lw=2, label=r'$\kappa$='+kstr)
ax.set_xlabel(r'$\lambda$', fontsize=12); ax.set_ylabel('G (absolute)', fontsize=12)
ax.legend(fontsize=10); ax.set_title(r'G magnitude scales with $\kappa$')
ax.set_yscale('log'); ax.set_ylim(1e-4, 1e3)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_kappa_sensitivity.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_kappa_sensitivity.png'), dpi=200, bbox_inches='tight')
plt.close(); print('saved fig_kappa_sensitivity')

#4: Finite-size scalingfig, ax = plt.subplots(figsize=(7, 5))
sizes = [('B75.0', 5, '#e377c2'), ('B168.75', 7.5, '#ff7f0e'),
         ('B300.0', 10, '#d62728'), ('B675.0', 15, '#1f77b4')]
for btag, L, c in sizes:
    lam, S_m, _, G_m, _, _, _, _ = agg(load(f'{btag}_k1.0_pxl0.5'))
    Gx = max(G_m.max(), 1e-10)
    ax.plot(lam, S_m, '--', color=c, alpha=0.4, lw=1.5)
    ax.plot(lam, np.clip(G_m/Gx, 0, None), '-s', color=c, ms=3, lw=2, label=f'L={L}')
ax.set_ylim(-0.05, 1.15); ax.set_xlabel(r'$\lambda$', fontsize=12)
ax.set_ylabel(r'S (dashed),  G/G$_{max}$ (solid)', fontsize=12)
ax.legend(fontsize=10, title='System size'); ax.set_title(r'Finite-size scaling (budget/L$^2$=3)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_finite_size.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_finite_size.png'), dpi=200, bbox_inches='tight')
plt.close(); print('saved fig_finite_size')

#5: B=1000 main resultlam, S_m, S_s, G_m, G_s, z_m, z_s, d_m = agg(load('B1000.0_k1.0_pxl0.5'))
Gx = max(G_m.max(), 1e-10); Gn = G_m/Gx; Gn_s = G_s/Gx

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
ax.fill_between(lam, np.clip(S_m-S_s,0,1), np.clip(S_m+S_s,0,1), alpha=0.15, color='#1f77b4')
ax.plot(lam, S_m, '-o', color='#1f77b4', ms=4, lw=2, label='S')
ax.fill_between(lam, np.clip(Gn-Gn_s,0,1.2), np.clip(Gn+Gn_s,0,1.2), alpha=0.15, color='#d62728')
ax.plot(lam, np.clip(Gn,0,None), '-s', color='#d62728', ms=4, lw=2, label=r'G/G$_{max}$')
ax.axvspan(0.38, 3.31, alpha=0.08, color='gold', label='gap')
ax.set_ylim(-0.05, 1.15); ax.set_xlabel(r'$\lambda$', fontsize=12)
ax.set_ylabel(r'S,  G/G$_{max}$', fontsize=12); ax.legend(fontsize=9)
ax.set_title('B=1000 (~1600 nodes/seed)')

ax = axes[1]
ax.fill_between(lam, z_m-z_s, z_m+z_s, alpha=0.15, color='k')
ax.plot(lam, z_m, 'k-o', ms=4, lw=2)
ax.axhline(4, color='red', ls='--', alpha=0.5, label='z=4')
ax.set_xlabel(r'$\lambda$', fontsize=12); ax.set_ylabel('2E/N', fontsize=12)
ax.legend(fontsize=9); ax.set_title('Coordination number')

ax = axes[2]
ax.plot(lam, d_m, '-o', color='#2ca02c', ms=4, lw=2)
ax.set_xlabel(r'$\lambda$', fontsize=12); ax.set_ylabel(r'nodes/L$^2$', fontsize=12)
ax.set_title('Intersection density')

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_main_B1000.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_main_B1000.png'), dpi=200, bbox_inches='tight')
plt.close(); print('saved fig_main_B1000')

#6: Network snapshotsfrom fiber_sim import FiberNetwork, plot_network
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (lv, label) in zip(axes, [(0.8, 'Disconnected'), (2.0, 'Connected, floppy'), (4.0, 'Connected, rigid')]):
    net = FiberNetwork(L=10, budget=300, lam=lv, p_xl=0.5, kappa=1.0, seed=0)
    net.build()
    plot_network(net, ax=ax, title=f'{label}\nλ={lv}, S={net.giant_component_frac():.2f}, z={net.mean_z():.1f}')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_network_snapshots.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_network_snapshots.png'), dpi=200, bbox_inches='tight')
plt.close(); print('saved fig_network_snapshots')

print(f'\nAll plots saved to:\n{OUT}')
