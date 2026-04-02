#!/usr/bin/env python3
"""Panel A: three connectivity levels via lambda sweep at B=3000, p_xl=0.5.
Produces energy density and structural (giant component) network plots."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fiber_sim import FiberNetwork

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'simulation_figure_panels')
os.makedirs(OUT, exist_ok=True)

B, L, PXL, KAPPA, K_SHAPE, GAMMA, SEED = 3000, 10, 0.5, 1.0, 4, 0.01, 0

# from lambda scan at B=3000, p_xl=0.5:
#   lam=0.15  N=14307  S=0.011  G=0.000  disconnected
#   lam=0.40  N=14251  S=0.995  G=0.071  gap (floppy, force chains)
#   lam=3.00  N=15001  S=1.000  G=0.554  rigid
CONFIGS = [
    (0.15, 'Disconnected'),
    (0.40, 'Connected (floppy)'),
    (3.0,  'Connected (rigid)'),
]


def build_and_shear(lam):
    net = FiberNetwork(L=L, budget=B, lam=lam, k_shape=K_SHAPE,
                       p_xl=PXL, kappa=KAPPA, seed=SEED)
    net.build()
    S = net.giant_component_frac()
    G = net.shear_modulus(GAMMA)
    pos_relax, edge_E = net.shear_edge_energies(GAMMA)

    row = np.concatenate([net.edges[:, 0], net.edges[:, 1]])
    col = np.concatenate([net.edges[:, 1], net.edges[:, 0]])
    Adj = csr_matrix((np.ones(len(row)), (row, col)),
                     shape=(net.n_nodes, net.n_nodes))
    _, comp_labels = connected_components(Adj, directed=False)
    gc_label = int(np.bincount(comp_labels).argmax())

    return {
        'lam': lam, 'S': S, 'G': G,
        'n_nodes': net.n_nodes, 'n_edges': net.n_edges,
        'pos': net.pos, 'pos_relax': pos_relax,
        'edges': net.edges, 'edge_E': edge_E,
        'starts': net.starts, 'ends': net.ends,
        'n_f': net.n_f, 'L': net.L,
        'comp_labels': comp_labels, 'gc_label': gc_label,
    }


def edge_segments(pos, edges, L):
    segs = []
    for ei in range(len(edges)):
        n1, n2 = edges[ei]
        p1 = pos[n1]
        p2 = pos[n2]
        d = p2 - p1
        d -= L * np.round(d / L)
        segs.append([(p1[0], p1[1]), (p1[0] + d[0], p1[1] + d[1])])
    return segs


lam_values = [c[0] for c in CONFIGS]
labels = [c[1] for c in CONFIGS]

print(f'Building 3 networks: B={B}, p_xl={PXL}, seed={SEED}')

with ProcessPoolExecutor(max_workers=3) as ex:
    results = list(tqdm(
        ex.map(build_and_shear, lam_values),
        total=3, desc='build+shear'))

for r, label in zip(results, labels):
    print(f'  {label}: lam={r["lam"]:.2f}  N={r["n_nodes"]}  E={r["n_edges"]}  '
          f'S={r["S"]:.3f}  G={r["G"]:.4f}')


# Energy figure
fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')

for i, (r, label) in enumerate(zip(results, labels)):
    ax = axes[i]
    ax.set_facecolor('white')

    esegs = edge_segments(r['pos'], r['edges'], r['L'])
    E = r['edge_E']
    E_max = max(E.max(), 1e-15)
    E_vis = np.sqrt(E / E_max)

    lw_min, lw_max = 0.2, 5.0
    lws = lw_min + (lw_max - lw_min) * E_vis

    colors = plt.cm.hot_r(E_vis)
    low = E_vis < 0.02
    colors[low] = [0.8, 0.8, 0.85, 0.4]
    lws[low] = 0.2

    ec = LineCollection(esegs, colors=colors, linewidths=lws)
    ax.add_collection(ec)

    ax.set_xlim(0, r['L']); ax.set_ylim(0, r['L'])
    ax.set_aspect('equal')
    ax.set_title(f'{label}\nlam={r["lam"]:.2f}  S={r["S"]:.2f}  '
                 f'G={r["G"]:.3f}  N={r["n_nodes"]}',
                 color='black', fontsize=13, pad=10)
    ax.axis('off')

plt.tight_layout()
for ext in ('pdf', 'png'):
    path = os.path.join(OUT, f'panel_A_energy_networks.{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


# Structural figure
fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')

for i, (r, label) in enumerate(zip(results, labels)):
    ax = axes2[i]
    ax.set_facecolor('white')

    esegs = edge_segments(r['pos'], r['edges'], r['L'])
    gc = r['gc_label']
    lbls = r['comp_labels']

    is_gc = np.array([lbls[r['edges'][ei, 0]] == gc
                      for ei in range(len(r['edges']))])
    edge_colors = np.where(
        is_gc[:, None],
        [0.12, 0.47, 0.71, 0.7],
        [0.84, 0.15, 0.16, 0.7],
    )

    ec = LineCollection(esegs, colors=edge_colors, linewidths=1.0)
    ax.add_collection(ec)

    z = 2 * r['n_edges'] / r['n_nodes']
    ax.set_xlim(0, r['L']); ax.set_ylim(0, r['L'])
    ax.set_aspect('equal')
    ax.set_title(f'{label}\nlam={r["lam"]:.2f}  S={r["S"]:.2f}  '
                 f'z={z:.1f}  N={r["n_nodes"]}',
                 color='black', fontsize=13, pad=10)
    ax.axis('off')

plt.tight_layout()
for ext in ('pdf', 'png'):
    path = os.path.join(OUT, f'panel_A_structural.{ext}')
    fig2.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
