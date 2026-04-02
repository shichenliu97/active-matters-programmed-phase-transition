"""
Fiber intersection network simulation.

N_f = budget/λ fibers with lengths ~ Gamma(k, λ/k) in L×L periodic box.
Geometric intersections → graph nodes. Consecutive intersections on same
fiber → edges. Central-force springs + position-based bending stiffness.

Key insight: fiber intersection graph naturally has z < 4 (fiber endpoints
reduce coordination). Central-force only → sub-isostatic → always floppy.
Bending stiffness adds constraints, enabling a connectivity-rigidity gap.

Bending model: for each triple (a,b,c) on a fiber, penalize b deviating
from the line a-c. E = (κ/2L)(dev)² where dev = distance of b from line ac.
This is numerically stable (no trig, no 1/r singularity).

Usage:
    python fiber_sim.py --validate
    python fiber_sim.py --sweep --budget 300 --kappa 1.0 --n_seeds 20
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, argparse, os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


class FiberNetwork:
    def __init__(self, L=10.0, budget=300, lam=1.0, k_shape=1,
                 p_xl=1.0, kappa=1.0, seed=0):
        self.L = L
        self.budget = budget
        self.lam = lam
        self.k_shape = k_shape
        self.p_xl = p_xl
        self.kappa = kappa
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.n_f = 0
        self.starts = self.ends = self.d_all = None
        self.xsects = []
        self.pos = np.empty((0, 2))
        self.edges = np.empty((0, 2), dtype=int)
        self.rest_L = np.empty(0)
        self.n_nodes = self.n_edges = 0
        self._fiber_xl = {}  # fiber_id → [(t_param, node_idx), ...]
        # bending: arrays for triples (a, b, c) on same fiber
        self._bend_a = np.empty(0, dtype=int)
        self._bend_b = np.empty(0, dtype=int)
        self._bend_c = np.empty(0, dtype=int)
        self._bend_w1 = np.empty(0)  # L1/(L1+L2)
        self._bend_w2 = np.empty(0)  # L2/(L1+L2)
        self._bend_Lt = np.empty(0)  # L1+L2

    def build(self):
        self._place_fibers()
        self._find_intersections()
        self._build_graph()
        return self

    def _place_fibers(self):
        n_f = max(1, int(self.budget / self.lam))
        self.n_f = n_f
        cx = self.rng.uniform(0, self.L, n_f)
        cy = self.rng.uniform(0, self.L, n_f)
        theta = self.rng.uniform(0, np.pi, n_f)
        flen = self.rng.gamma(self.k_shape, self.lam / self.k_shape, n_f)

        dx = flen / 2 * np.cos(theta)
        dy = flen / 2 * np.sin(theta)
        self.starts = np.column_stack([cx - dx, cy - dy])
        self.ends = np.column_stack([cx + dx, cy + dy])
        self.d_all = self.ends - self.starts

    def _find_intersections(self):
        n_f = self.n_f
        L = self.L
        if n_f < 2:
            self.xsects = []
            return

        ii, jj = np.triu_indices(n_f, k=1)
        n_pairs = len(ii)

        p1 = self.starts[ii]
        d1 = self.d_all[ii]
        d2 = self.d_all[jj]
        sj = self.starts[jj]

        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        ok = np.abs(cross) > 1e-12
        inv_cross = np.zeros(n_pairs)
        inv_cross[ok] = 1.0 / cross[ok]

        xsects = []
        for sdx in (-1, 0, 1):
            for sdy in (-1, 0, 1):
                p3 = sj + np.array([sdx * L, sdy * L])
                dp = p3 - p1

                t = np.full(n_pairs, -1.0)
                u = np.full(n_pairs, -1.0)
                t[ok] = (dp[ok, 0] * d2[ok, 1] - dp[ok, 1] * d2[ok, 0]) * inv_cross[ok]
                u[ok] = (dp[ok, 0] * d1[ok, 1] - dp[ok, 1] * d1[ok, 0]) * inv_cross[ok]

                hit = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
                idx = np.where(hit)[0]
                if len(idx) == 0:
                    continue

                pts = p1[idx] + t[idx, None] * d1[idx]
                pts %= L
                for k, m in enumerate(idx):
                    xsects.append((ii[m], jj[m], t[m], u[m],
                                   pts[k, 0], pts[k, 1]))

        seen = {}
        for x in xsects:
            key = (x[0], x[1])
            if key not in seen:
                seen[key] = x
        self.xsects = list(seen.values())

    def _build_graph(self):
        xsects = self.xsects
        if not xsects:
            self._empty_graph()
            return

        if self.p_xl < 1.0:
            keep = self.rng.random(len(xsects)) < self.p_xl
            xsects = [x for x, k in zip(xsects, keep) if k]
        if not xsects:
            self._empty_graph()
            return

        self.pos = np.array([(x[4], x[5]) for x in xsects])
        self.n_nodes = len(self.pos)

        fiber_xl = {}
        for idx, (fi, fj, ti, uj, _, _) in enumerate(xsects):
            fiber_xl.setdefault(fi, []).append((ti, idx))
            fiber_xl.setdefault(fj, []).append((uj, idx))

        self._fiber_xl = fiber_xl

        edges = []
        ba, bb, bc, bL1, bL2 = [], [], [], [], []

        for pts in fiber_xl.values():
            pts.sort()
            for k in range(len(pts) - 1):
                edges.append((pts[k][1], pts[k + 1][1]))
            # bending triples
            for k in range(len(pts) - 2):
                ai, bi, ci = pts[k][1], pts[k + 1][1], pts[k + 2][1]
                d1 = self.pos[bi] - self.pos[ai]
                d1 -= self.L * np.round(d1 / self.L)
                L1 = np.linalg.norm(d1)
                d2 = self.pos[ci] - self.pos[bi]
                d2 -= self.L * np.round(d2 / self.L)
                L2 = np.linalg.norm(d2)
                if L1 > 1e-10 and L2 > 1e-10:
                    ba.append(ai); bb.append(bi); bc.append(ci)
                    bL1.append(L1); bL2.append(L2)

        if not edges:
            self._empty_graph()
            return

        self.edges = np.array(edges, dtype=int)
        self.n_edges = len(self.edges)

        d = self.pos[self.edges[:, 1]] - self.pos[self.edges[:, 0]]
        d -= self.L * np.round(d / self.L)
        self.rest_L = np.linalg.norm(d, axis=1)

        mask = self.rest_L > 1e-10
        self.edges = self.edges[mask]
        self.rest_L = self.rest_L[mask]
        self.n_edges = len(self.edges)

        # store bending data
        if ba:
            self._bend_a = np.array(ba, dtype=int)
            self._bend_b = np.array(bb, dtype=int)
            self._bend_c = np.array(bc, dtype=int)
            L1a = np.array(bL1)
            L2a = np.array(bL2)
            self._bend_Lt = L1a + L2a
            self._bend_w1 = L1a / self._bend_Lt
            self._bend_w2 = L2a / self._bend_Lt
        else:
            self._bend_a = np.empty(0, dtype=int)
            self._bend_b = np.empty(0, dtype=int)
            self._bend_c = np.empty(0, dtype=int)
            self._bend_w1 = np.empty(0)
            self._bend_w2 = np.empty(0)
            self._bend_Lt = np.empty(0)

    def _empty_graph(self):
        self.pos = np.empty((0, 2))
        self.edges = np.empty((0, 2), dtype=int)
        self.rest_L = np.empty(0)
        self.n_nodes = self.n_edges = 0
        self._fiber_xl = {}
        self._bend_a = np.empty(0, dtype=int)
        self._bend_b = np.empty(0, dtype=int)
        self._bend_c = np.empty(0, dtype=int)
        self._bend_w1 = np.empty(0)
        self._bend_w2 = np.empty(0)
        self._bend_Lt = np.empty(0)

    # ── topology ──

    def giant_component_frac(self):
        if self.n_nodes < 2:
            return 0.0
        if self.n_edges == 0:
            return 1.0 / self.n_nodes
        row = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        col = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
        A = csr_matrix((np.ones(len(row)), (row, col)),
                       shape=(self.n_nodes, self.n_nodes))
        _, labels = connected_components(A, directed=False)
        return int(np.bincount(labels).max()) / self.n_nodes

    def mean_z(self):
        if self.n_nodes == 0:
            return 0.0
        return 2 * self.n_edges / self.n_nodes

    def n_components(self):
        """Number of connected components."""
        if self.n_nodes < 2 or self.n_edges == 0:
            return self.n_nodes
        row = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        col = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
        A = csr_matrix((np.ones(len(row)), (row, col)),
                       shape=(self.n_nodes, self.n_nodes))
        n_comp, _ = connected_components(A, directed=False)
        return n_comp

    def crosslinks_per_fiber(self):
        """Number of crosslinks per fiber (fibers with ≥1 crosslink only)."""
        if not self._fiber_xl:
            return np.array([], dtype=int)
        return np.array([len(v) for v in self._fiber_xl.values()], dtype=int)

    def betti_1(self):
        """First Betti number: independent cycles = E - N + n_components."""
        if self.n_nodes < 2 or self.n_edges == 0:
            return 0
        row = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        col = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
        A = csr_matrix((np.ones(len(row)), (row, col)),
                       shape=(self.n_nodes, self.n_nodes))
        n_comp, _ = connected_components(A, directed=False)
        return self.n_edges - self.n_nodes + n_comp

    # ── mechanics ──

    def shear_modulus(self, gamma=0.01):
        """Energy-based G with Lees-Edwards PBC (no fixed walls).
        G = (E(+γ)+E(-γ)) / (γ²A). Always non-negative."""
        if self.n_nodes < 4 or self.n_edges < 4:
            return 0.0

        def shear_energy(g):
            p = self.pos.copy()
            p[:, 0] += g * (p[:, 1] - self.L / 2)
            p = self._fire_le(p, g)
            return self._energy_le(p, g)

        E_p = shear_energy(gamma)
        E_m = shear_energy(-gamma)
        return (E_p + E_m) / (gamma**2 * self.L**2)

    def shear_edge_energies(self, gamma=0.01):
        """Apply shear, FIRE-relax, return per-edge stretch energies."""
        if self.n_nodes < 4 or self.n_edges < 4:
            return self.pos.copy(), np.zeros(self.n_edges)
        p = self.pos.copy()
        p[:, 0] += gamma * (p[:, 1] - self.L / 2)
        p = self._fire_le(p, gamma)
        i, j = self.edges[:, 0], self.edges[:, 1]
        d = (p[j] - p[i]).copy()
        d = self._min_image_le(d, gamma)
        dist = np.linalg.norm(d, axis=1)
        edge_E = 0.5 * (dist - self.rest_L)**2
        return p, edge_E

    def _shear_relax(self, gamma):
        """Apply affine shear and FIRE-relax. Returns (relaxed_pos, affine_pos)."""
        p_aff = self.pos.copy()
        p_aff[:, 0] += gamma * (p_aff[:, 1] - self.L / 2)
        p_relax = self._fire_le(p_aff.copy(), gamma)
        return p_relax, p_aff

    def energy_components(self, gamma=0.01):
        """Return (E_stretch, E_bend) after shear+relax."""
        if self.n_nodes < 4 or self.n_edges < 4:
            return 0.0, 0.0
        p, _ = self._shear_relax(gamma)
        E_s = 0.0
        if self.n_edges > 0:
            i, j = self.edges[:, 0], self.edges[:, 1]
            d = (p[j] - p[i]).copy()
            d = self._min_image_le(d, gamma)
            dist = np.linalg.norm(d, axis=1)
            E_s = 0.5 * np.sum((dist - self.rest_L)**2)
        E_b = 0.0
        if self.kappa > 0 and len(self._bend_a) > 0:
            a, b, c = self._bend_a, self._bend_b, self._bend_c
            d_ba = (p[a] - p[b]).copy()
            d_bc = (p[c] - p[b]).copy()
            self._min_image_le(d_ba, gamma)
            self._min_image_le(d_bc, gamma)
            dev = -(self._bend_w2[:, None] * d_ba + self._bend_w1[:, None] * d_bc)
            E_b = 0.5 * self.kappa * np.sum(np.sum(dev**2, axis=1) / self._bend_Lt)
        return float(E_s), float(E_b)

    def non_affinity(self, gamma=0.01):
        """Non-affinity parameter Gamma = <|delta_na|^2> / <|delta_aff|^2>."""
        if self.n_nodes < 4 or self.n_edges < 4:
            return 0.0
        p_relax, p_aff = self._shear_relax(gamma)
        delta_aff = p_aff - self.pos
        delta_na = p_relax - p_aff
        aff_sq = np.mean(np.sum(delta_aff**2, axis=1))
        na_sq = np.mean(np.sum(delta_na**2, axis=1))
        if aff_sq < 1e-30:
            return 0.0
        return float(na_sq / aff_sq)

    def participation_ratio(self, gamma=0.01):
        """PR = (sum e_i)^2 / (N * sum e_i^2). Fraction of edges carrying load."""
        if self.n_nodes < 4 or self.n_edges < 4:
            return 0.0
        _, edge_E = self.shear_edge_energies(gamma)
        E_sum = edge_E.sum()
        E2_sum = (edge_E**2).sum()
        if E2_sum < 1e-30:
            return 0.0
        return float(E_sum**2 / (len(edge_E) * E2_sum))

    def full_measure(self, gamma=0.01):
        """Complete measurement dict for dimensionless sweep."""
        S = self.giant_component_frac()
        z = self.mean_z()
        n_comp = self.n_components()
        beta1 = self.n_edges - self.n_nodes + n_comp if self.n_nodes > 1 else 0
        k_per = self.crosslinks_per_fiber()
        k_mean = float(k_per.mean()) if len(k_per) > 0 else 0.0
        k_std = float(k_per.std()) if len(k_per) > 0 else 0.0
        k_dist = np.bincount(k_per).tolist() if len(k_per) > 0 else []
        n_active = len(self._fiber_xl)
        f_iso = 1.0 - n_active / self.n_f if self.n_f > 0 else 1.0

        G = self.shear_modulus(gamma) if self.n_nodes > 10 else 0.0
        E_s, E_b = self.energy_components(gamma) if self.n_nodes > 10 else (0.0, 0.0)
        Gamma = self.non_affinity(gamma) if self.n_nodes > 10 else 0.0
        PR = self.participation_ratio(gamma) if self.n_nodes > 10 else 0.0

        return {
            'S': float(S), 'z': float(z), 'G': float(G),
            'N_nodes': self.n_nodes, 'N_edges': self.n_edges,
            'N_components': n_comp, 'beta1': beta1,
            'k_mean': k_mean, 'k_std': k_std, 'k_distribution': k_dist,
            'N_fibers_total': self.n_f, 'N_fibers_with_xl': n_active,
            'f_isolated': float(f_iso),
            'E_total': E_s + E_b, 'E_stretch': E_s, 'E_bend': E_b,
            'Gamma': Gamma, 'PR': PR,
        }

    # ── Lees-Edwards shear mechanics ──

    def _min_image_le(self, d, gamma):
        """Lees-Edwards minimum image: y-wrap also shifts x by γL."""
        n_y = np.round(d[:, 1] / self.L)
        d[:, 1] -= self.L * n_y
        d[:, 0] -= gamma * self.L * n_y
        d[:, 0] -= self.L * np.round(d[:, 0] / self.L)
        return d

    def _fire_le(self, pos, gamma, max_iter=10000, ftol=1e-8):
        """FIRE with Lees-Edwards PBC, all nodes free."""
        dt, dt_max = 0.005, 0.05
        alpha, a0, fa = 0.1, 0.1, 0.99
        n_pos_steps = 0
        vel = np.zeros_like(pos)
        p = pos.copy()

        for _ in range(max_iter):
            F = self._forces_le(p, gamma)

            fmax = np.sqrt((F * F).sum(axis=1).max())
            if fmax < ftol:
                break

            P = (F * vel).sum()
            if P > 0:
                n_pos_steps += 1
                vnorm = np.sqrt((vel * vel).sum())
                fnorm = np.sqrt((F * F).sum())
                if fnorm > 1e-20:
                    vel = (1 - alpha) * vel + alpha * (vnorm / fnorm) * F
                if n_pos_steps > 5:
                    dt = min(dt * 1.1, dt_max)
                    alpha *= fa
            else:
                vel[:] = 0.0
                dt *= 0.5
                alpha = a0
                n_pos_steps = 0

            vel += F * dt
            p += vel * dt

        return p

    def _forces_le(self, pos, gamma):
        """Total forces with Lees-Edwards PBC."""
        F = np.zeros_like(pos)
        if self.n_edges > 0:
            i, j = self.edges[:, 0], self.edges[:, 1]
            d = pos[j] - pos[i]
            d = self._min_image_le(d, gamma)
            dist = np.linalg.norm(d, axis=1)
            ok = dist > 1e-12
            mag = np.zeros(self.n_edges)
            mag[ok] = (dist[ok] - self.rest_L[ok]) / dist[ok]
            fx = mag * d[:, 0]
            fy = mag * d[:, 1]
            np.add.at(F[:, 0], i, fx)
            np.add.at(F[:, 1], i, fy)
            np.add.at(F[:, 0], j, -fx)
            np.add.at(F[:, 1], j, -fy)

        if self.kappa > 0 and len(self._bend_a) > 0:
            a, b, c = self._bend_a, self._bend_b, self._bend_c
            d_ba = (pos[a] - pos[b]).copy()
            d_bc = (pos[c] - pos[b]).copy()
            self._min_image_le(d_ba, gamma)
            self._min_image_le(d_bc, gamma)
            dev = -(self._bend_w2[:, None] * d_ba + self._bend_w1[:, None] * d_bc)
            pref = self.kappa / self._bend_Lt
            fb = -pref[:, None] * dev
            fa = (self._bend_w2 * pref)[:, None] * dev
            fc = (self._bend_w1 * pref)[:, None] * dev
            for dim in range(2):
                np.add.at(F[:, dim], a, fa[:, dim])
                np.add.at(F[:, dim], b, fb[:, dim])
                np.add.at(F[:, dim], c, fc[:, dim])

        return F

    def _energy_le(self, pos, gamma):
        """Total energy with Lees-Edwards PBC."""
        E = 0.0
        if self.n_edges > 0:
            i, j = self.edges[:, 0], self.edges[:, 1]
            d = pos[j] - pos[i]
            d = self._min_image_le(d, gamma)
            dist = np.linalg.norm(d, axis=1)
            E += 0.5 * np.sum((dist - self.rest_L)**2)
        if self.kappa > 0 and len(self._bend_a) > 0:
            a, b, c = self._bend_a, self._bend_b, self._bend_c
            d_ba = (pos[a] - pos[b]).copy()
            d_bc = (pos[c] - pos[b]).copy()
            self._min_image_le(d_ba, gamma)
            self._min_image_le(d_bc, gamma)
            dev = -(self._bend_w2[:, None] * d_ba + self._bend_w1[:, None] * d_bc)
            E += 0.5 * self.kappa * np.sum(np.sum(dev**2, axis=1) / self._bend_Lt)
        return E

    def _fire(self, pos, fixed, max_iter=10000, ftol=1e-8):
        dt, dt_max = 0.005, 0.05
        alpha, a0, fa = 0.1, 0.1, 0.99
        n_pos_steps = 0
        vel = np.zeros_like(pos)
        p = pos.copy()

        for _ in range(max_iter):
            F = self._total_forces(p)
            F[fixed] = 0.0

            fmax = np.sqrt((F * F).sum(axis=1).max())
            if fmax < ftol:
                break

            P = (F * vel).sum()
            if P > 0:
                n_pos_steps += 1
                vnorm = np.sqrt((vel * vel).sum())
                fnorm = np.sqrt((F * F).sum())
                if fnorm > 1e-20:
                    vel = (1 - alpha) * vel + alpha * (vnorm / fnorm) * F
                if n_pos_steps > 5:
                    dt = min(dt * 1.1, dt_max)
                    alpha *= fa
            else:
                vel[:] = 0.0
                dt *= 0.5
                alpha = a0
                n_pos_steps = 0

            vel += F * dt
            vel[fixed] = 0.0
            p += vel * dt

        return p

    def _total_forces(self, pos):
        F = self._stretch_forces(pos)
        if self.kappa > 0 and len(self._bend_a) > 0:
            F += self._bending_forces(pos)
        return F

    def _stretch_forces(self, pos):
        F = np.zeros_like(pos)
        if self.n_edges == 0:
            return F
        i, j = self.edges[:, 0], self.edges[:, 1]
        d = pos[j] - pos[i]
        d -= self.L * np.round(d / self.L)
        dist = np.linalg.norm(d, axis=1)

        ok = dist > 1e-12
        mag = np.zeros(self.n_edges)
        mag[ok] = (dist[ok] - self.rest_L[ok]) / dist[ok]

        fx = mag * d[:, 0]
        fy = mag * d[:, 1]
        np.add.at(F[:, 0], i, fx)
        np.add.at(F[:, 1], i, fy)
        np.add.at(F[:, 0], j, -fx)
        np.add.at(F[:, 1], j, -fy)
        return F

    def _bending_forces(self, pos):
        """Position-based bending: E = (κ/2Lt)|dev|² where dev measures
        deviation of b from line a-c. Numerically stable, no trig."""
        F = np.zeros_like(pos)
        a, b, c = self._bend_a, self._bend_b, self._bend_c

        d_ba = pos[a] - pos[b]
        d_bc = pos[c] - pos[b]
        d_ba -= self.L * np.round(d_ba / self.L)
        d_bc -= self.L * np.round(d_bc / self.L)

        # dev = -(w2*d_ba + w1*d_bc) = 0 when collinear
        dev = -(self._bend_w2[:, None] * d_ba + self._bend_w1[:, None] * d_bc)

        pref = self.kappa / self._bend_Lt  # (n_triples,)

        fb = -pref[:, None] * dev
        fa = (self._bend_w2 * pref)[:, None] * dev
        fc = (self._bend_w1 * pref)[:, None] * dev

        for dim in range(2):
            np.add.at(F[:, dim], a, fa[:, dim])
            np.add.at(F[:, dim], b, fb[:, dim])
            np.add.at(F[:, dim], c, fc[:, dim])

        return F

    # ── measure ──

    def measure(self, compute_G=True):
        S = self.giant_component_frac()
        z = self.mean_z()
        G = self.shear_modulus() if (compute_G and self.n_nodes > 10) else 0.0
        return {
            'lam': self.lam, 'seed': self.seed, 'p_xl': self.p_xl,
            'kappa': self.kappa,
            'S': S, 'G': G, 'z': z,
            'n_nodes': self.n_nodes, 'n_edges': self.n_edges,
            'n_fibers': self.n_f, 'n_xsects': len(self.xsects),
            'n_triples': len(self._bend_a),
            'beta_1': self.betti_1(),
            'density': self.n_nodes / self.L**2,
        }


# ── Parallel sweep ───────────────────────────────────────────────

def _run_one(args):
    lam, seed, L, budget, k_shape, p_xl, kappa, compute_G = args
    net = FiberNetwork(L=L, budget=budget, lam=lam, k_shape=k_shape,
                       p_xl=p_xl, kappa=kappa, seed=seed)
    net.build()
    return net.measure(compute_G=compute_G)


def sweep(lam_range, n_seeds=20, L=10.0, budget=300, k_shape=1,
          p_xl=1.0, kappa=1.0, n_workers=64, compute_G=True):
    tasks = [(lam, s, L, budget, k_shape, p_xl, kappa, compute_G)
             for lam in lam_range for s in range(n_seeds)]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for r in tqdm(pool.map(_run_one, tasks, chunksize=max(1, len(tasks) // n_workers)),
                      total=len(tasks), desc=f'κ={kappa} p_xl={p_xl}'):
            results.append(r)
    return results


# ── Analysis ─────────────────────────────────────────────────────

def find_thresholds(results):
    import pandas as pd
    df = pd.DataFrame(results)
    agg = df.groupby('lam').agg(['mean', 'std'])

    lam = agg.index.values
    S = agg[('S', 'mean')].values
    G = agg[('G', 'mean')].values
    z = agg[('z', 'mean')].values

    lam_conn = None
    for k in range(len(lam) - 1):
        if S[k] < 0.5 <= S[k + 1]:
            f = (0.5 - S[k]) / (S[k + 1] - S[k])
            lam_conn = lam[k] + f * (lam[k + 1] - lam[k])
            break

    lam_rigid = None
    G_max = G.max()
    if G_max > 0:
        G_norm = G / G_max
        for k in range(len(lam) - 1):
            if G_norm[k] < 0.1 <= G_norm[k + 1]:
                f = (0.1 - G_norm[k]) / (G_norm[k + 1] - G_norm[k])
                lam_rigid = lam[k] + f * (lam[k + 1] - lam[k])
                break

    gap = None
    if lam_conn is not None and lam_rigid is not None:
        gap = lam_rigid - lam_conn

    return {
        'lam_conn': lam_conn, 'lam_rigid': lam_rigid, 'gap': gap,
        'G_max': float(G_max) if G_max > 0 else 0.0,
        'z_mean': float(z.mean()), 'z_std': float(z.std()),
    }


# ── Plotting ─────────────────────────────────────────────────────

def plot_sweep(results, title='', save=None):
    import pandas as pd
    df = pd.DataFrame(results)
    agg = df.groupby('lam').agg(['mean', 'std'])

    lam = agg.index.values
    S_m, S_s = agg[('S', 'mean')].values, agg[('S', 'std')].values
    z_m, z_s = agg[('z', 'mean')].values, agg[('z', 'std')].values
    d_m, d_s = agg[('density', 'mean')].values, agg[('density', 'std')].values
    G_m, G_s = agg[('G', 'mean')].values, agg[('G', 'std')].values

    G_max = max(G_m.max(), 1e-10)
    G_norm = G_m / G_max
    G_norm_s = G_s / G_max

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.fill_between(lam, np.clip(S_m - S_s, 0, None), np.clip(S_m + S_s, None, 1.05),
                    alpha=0.15, color='C0')
    ax.plot(lam, S_m, 'C0-o', ms=3, label='S')
    ax.fill_between(lam, np.clip(G_norm - G_norm_s, 0, None),
                    np.clip(G_norm + G_norm_s, None, 1.5), alpha=0.15, color='C1')
    ax.plot(lam, np.clip(G_norm, 0, None), 'C1-s', ms=3, label='G/G_max')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.4)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('λ (mean fiber length)')
    ax.set_ylabel('S, G/G_max')
    ax.legend(loc='center right')
    ax.set_title('Connectivity & Rigidity')

    ax = axes[0, 1]
    ax.fill_between(lam, z_m - z_s, z_m + z_s, alpha=0.15, color='k')
    ax.plot(lam, z_m, 'k-o', ms=3)
    ax.axhline(4, color='r', ls='--', alpha=0.5, label='z=4 (Maxwell)')
    ax.set_xlabel('λ')
    ax.set_ylabel('2E/N')
    ax.legend()
    ax.set_title('Coordination number')

    ax = axes[1, 0]
    ax.fill_between(lam, d_m - d_s, d_m + d_s, alpha=0.15, color='C2')
    ax.plot(lam, d_m, 'C2-o', ms=3)
    ax.set_xlabel('λ')
    ax.set_ylabel('nodes / L²')
    ax.set_title('Intersection density')

    ax = axes[1, 1]
    nn_m = agg[('n_nodes', 'mean')].values
    ne_m = agg[('n_edges', 'mean')].values
    ax.plot(lam, nn_m, 'C3-o', ms=3, label='nodes')
    ax.plot(lam, ne_m, 'C4-s', ms=3, label='edges')
    ax.set_xlabel('λ')
    ax.set_ylabel('count')
    ax.legend()
    ax.set_title('Graph size')

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f'  saved {save}')
    plt.close(fig)


def plot_network(net, ax=None, title=''):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))
    L = net.L
    for k in range(net.n_f):
        s, e = net.starts[k] % L, net.ends[k] % L
        ax.plot([s[0], e[0]], [s[1], e[1]], 'k-', alpha=0.15, lw=0.5)

    if net.n_edges > 0 and net.n_nodes > 1:
        row = np.concatenate([net.edges[:, 0], net.edges[:, 1]])
        col = np.concatenate([net.edges[:, 1], net.edges[:, 0]])
        Adj = csr_matrix((np.ones(len(row)), (row, col)),
                         shape=(net.n_nodes, net.n_nodes))
        _, labels = connected_components(Adj, directed=False)
        gc_label = np.bincount(labels).argmax()

        for ei in range(net.n_edges):
            n1, n2 = net.edges[ei]
            p1, p2 = net.pos[n1], net.pos[n2]
            d = p2 - p1
            d -= L * np.round(d / L)
            c = 'C0' if labels[n1] == gc_label else 'C3'
            ax.plot([p1[0], p1[0] + d[0]], [p1[1], p1[1] + d[1]],
                    c=c, lw=1.0, alpha=0.7)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_title(title)
    return ax


# ── Validation ───────────────────────────────────────────────────

def validate():
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        if cond:
            print(f'  ✓ {name}')
        else:
            print(f'  ✗ {name} — {detail}')
            ok = False

    print('\n── Step 1: Segment intersection ──')
    net = FiberNetwork(L=10, budget=100, lam=1, seed=0)
    net.n_f = 2
    net.starts = np.array([[0.0, 5.0], [5.0, 0.0]])
    net.ends = np.array([[10.0, 5.0], [5.0, 10.0]])
    net.d_all = net.ends - net.starts
    net._find_intersections()
    check('2 crossing fibers → 1 intersection', len(net.xsects) == 1,
          f'got {len(net.xsects)}')
    if net.xsects:
        pt = np.array([net.xsects[0][4], net.xsects[0][5]])
        check('intersection at (5,5)', np.allclose(pt, [5, 5], atol=0.1),
              f'got {pt}')

    net.n_f = 3
    net.starts = np.array([[2., 3.], [8., 3.], [5., 2.]])
    net.ends = np.array([[8., 7.], [2., 7.], [5., 8.]])
    net.d_all = net.ends - net.starts
    net._find_intersections()
    check('3 crossing fibers → 3 intersections', len(net.xsects) == 3,
          f'got {len(net.xsects)}')

    print('\n── Step 2: Graph construction ──')
    net2 = FiberNetwork(L=10, budget=300, lam=1.5, seed=42, kappa=0)
    net2.build()
    z = net2.mean_z()
    check(f'z in [2.5, 4.2]: z={z:.2f}', 2.5 < z < 4.2, f'z = {z:.2f}')
    check(f'has nodes: {net2.n_nodes}', net2.n_nodes > 10)
    check(f'has edges: {net2.n_edges}', net2.n_edges > 10)

    net3 = FiberNetwork(L=10, budget=300, lam=1.5, p_xl=0.5, seed=42, kappa=0)
    net3.build()
    z3 = net3.mean_z()
    check(f'z with p_xl=0.5: z={z3:.2f} ≤ z_full+0.5', z3 <= z + 0.5)

    print('\n── Step 3: Connectivity ──')
    lams = np.linspace(0.3, 4.0, 15)
    S_vals = []
    for lam in lams:
        net_t = FiberNetwork(L=10, budget=300, lam=lam, seed=0, kappa=0)
        net_t.build()
        S_vals.append(net_t.giant_component_frac())
    S_vals = np.array(S_vals)
    check(f'S rises: {S_vals[0]:.2f} → {S_vals[-1]:.2f}',
          S_vals[-1] > S_vals[0] + 0.3)
    check('S > 0.8 at large λ', S_vals[-1] > 0.8, f'{S_vals[-1]:.2f}')

    print('\n── Step 4: Bending force sanity ──')
    # 3 collinear nodes: bending force should be zero
    net4 = FiberNetwork(L=10, budget=100, lam=1, kappa=1.0, seed=0)
    net4.pos = np.array([[2.0, 5.0], [5.0, 5.0], [8.0, 5.0]])
    net4.n_nodes = 3
    net4.edges = np.array([[0, 1], [1, 2]])
    net4.rest_L = np.array([3.0, 3.0])
    net4.n_edges = 2
    net4._bend_a = np.array([0])
    net4._bend_b = np.array([1])
    net4._bend_c = np.array([2])
    net4._bend_Lt = np.array([6.0])
    net4._bend_w1 = np.array([0.5])
    net4._bend_w2 = np.array([0.5])
    F_straight = net4._bending_forces(net4.pos)
    check('bending force = 0 for straight fiber',
          np.allclose(F_straight, 0, atol=1e-10),
          f'max |F| = {np.abs(F_straight).max():.2e}')

    # bend middle node: should get restoring force
    bent_pos = net4.pos.copy()
    bent_pos[1, 1] += 0.5  # push middle up
    F_bent = net4._bending_forces(bent_pos)
    check('bending force pushes bent node back (F_y < 0)',
          F_bent[1, 1] < 0,
          f'F_b_y = {F_bent[1, 1]:.4f}')
    check('bending forces sum to zero',
          np.allclose(F_bent.sum(axis=0), 0, atol=1e-10),
          f'sum = {F_bent.sum(axis=0)}')

    print('\n── Step 5: Shear modulus (Lees-Edwards energy) ──')
    net_r = FiberNetwork(L=10, budget=300, lam=3.0, seed=0, kappa=1.0)
    net_r.build()
    G = net_r.shear_modulus()
    print(f'  info: κ=1, λ=3: G={G:.4f}, z={net_r.mean_z():.2f}, '
          f'S={net_r.giant_component_frac():.2f}, triples={len(net_r._bend_a)}')
    check(f'G > 0 with bending: G={G:.4f}', G > 0.01, f'G = {G}')

    net_s = FiberNetwork(L=10, budget=300, lam=0.3, seed=0, kappa=1.0)
    net_s.build()
    G_s = net_s.shear_modulus()
    print(f'  info: κ=1, λ=0.3 (sparse): G={G_s:.6f}, S={net_s.giant_component_frac():.2f}')
    check(f'G ≈ 0 for sparse/disconnected: G={G_s:.4f}',
          G_s < 0.1, f'G = {G_s}')

    net_cf = FiberNetwork(L=10, budget=300, lam=3.0, seed=0, kappa=0)
    net_cf.build()
    G_cf = net_cf.shear_modulus()
    print(f'  info: κ=0, λ=3: G={G_cf:.6f}')
    check(f'G(κ=1) > G(κ=0): {G:.4f} vs {G_cf:.4f}', G > G_cf,
          f'G_bend={G:.4f}, G_cf={G_cf:.4f}')

    print(f'\n{"=" * 50}')
    print(f'{"VALIDATION PASSED" if ok else "VALIDATION FAILED"}')
    print(f'{"=" * 50}')
    return ok


# ── Main ─────────────────────────────────────────────────────────

def run_sweep(args):
    lam_range = np.linspace(args.lam_min, args.lam_max, args.n_lam)
    tag = f'B{args.budget}_k{args.kappa}_pxl{args.p_xl}'

    print(f'\n{"=" * 60}')
    print(f'Sweep: budget={args.budget}, κ={args.kappa}, p_xl={args.p_xl}, '
          f'λ=[{args.lam_min:.1f},{args.lam_max:.1f}], '
          f'{args.n_lam} pts × {args.n_seeds} seeds')
    print(f'Workers: {args.n_workers}, compute_G: {not args.skip_G}')
    print(f'{"=" * 60}')

    results = sweep(lam_range, n_seeds=args.n_seeds, L=args.L,
                    budget=args.budget, k_shape=args.k_shape,
                    p_xl=args.p_xl, kappa=args.kappa,
                    n_workers=args.n_workers,
                    compute_G=not args.skip_G)

    json_path = os.path.join(OUT_DIR, f'sweep_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f)
    print(f'  saved {json_path}')

    png_path = os.path.join(OUT_DIR, f'sweep_{tag}.png')
    plot_sweep(results, title=tag, save=png_path)

    th = find_thresholds(results)
    print(f'\n  λ_conn  = {th["lam_conn"]}')
    print(f'  λ_rigid = {th["lam_rigid"]}')
    print(f'  gap     = {th["gap"]}')
    print(f'  G_max   = {th["G_max"]:.6f}')
    print(f'  z_mean  = {th["z_mean"]:.2f} ± {th["z_std"]:.2f}')

    if th['gap'] is not None and th['gap'] > 0:
        print(f'\n  ★ GAP EXISTS: Δλ = {th["gap"]:.3f} ★')
    elif th['lam_conn'] is not None and th['lam_rigid'] is None:
        print(f'\n  ⚠ S transition found but no G transition')
    elif th['gap'] is not None and th['gap'] <= 0:
        print(f'\n  ⚠ NO GAP — G rises with S')
    else:
        print(f'\n  ⚠ Thresholds not found — adjust λ range or budget')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    snap_lams = [lam_range[len(lam_range) // 6],
                 lam_range[len(lam_range) // 2],
                 lam_range[5 * len(lam_range) // 6]]
    for ax, sl in zip(axes, snap_lams):
        net = FiberNetwork(L=args.L, budget=args.budget, lam=sl,
                           k_shape=args.k_shape, p_xl=args.p_xl,
                           kappa=args.kappa, seed=0)
        net.build()
        plot_network(net, ax=ax,
                     title=f'λ={sl:.2f}, S={net.giant_component_frac():.2f}, '
                           f'z={net.mean_z():.1f}')
    fig.tight_layout()
    snap_path = os.path.join(OUT_DIR, f'networks_{tag}.png')
    fig.savefig(snap_path, dpi=150, bbox_inches='tight')
    print(f'  saved {snap_path}')
    plt.close(fig)

    return results, th


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validate', action='store_true')
    p.add_argument('--sweep', action='store_true')
    p.add_argument('--budget', type=float, default=300)
    p.add_argument('--kappa', type=float, default=1.0)
    p.add_argument('--p_xl', type=float, default=1.0)
    p.add_argument('--L', type=float, default=10.0)
    p.add_argument('--k_shape', type=float, default=1.0)
    p.add_argument('--lam_min', type=float, default=0.3)
    p.add_argument('--lam_max', type=float, default=5.0)
    p.add_argument('--n_lam', type=int, default=30)
    p.add_argument('--n_seeds', type=int, default=20)
    p.add_argument('--n_workers', type=int, default=64)
    p.add_argument('--skip_G', action='store_true')
    args = p.parse_args()

    if args.validate:
        validate()
    elif args.sweep:
        run_sweep(args)
    else:
        if validate():
            args.n_seeds = 10
            args.n_lam = 20
            run_sweep(args)


if __name__ == '__main__':
    main()
