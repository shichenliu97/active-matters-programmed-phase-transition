#!/usr/bin/env python3
"""Displacement correlation C(r,tau) from PIV velocity data."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pickle
import os
from tqdm import tqdm


def first_crossing(r, y, threshold=1/np.e):
    """Return r where y first crosses below threshold (linear interpolation)."""
    y = np.asarray(y)
    r = np.asarray(r)
    ok = np.isfinite(y) & np.isfinite(r)
    y = y[ok]; r = r[ok]
    idx = np.where(y < threshold)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return r[0]
    r0, r1 = r[i-1], r[i]
    y0, y1 = y[i-1], y[i]
    if y1 == y0:
        return r1
    return r0 + (threshold - y0) * (r1 - r0) / (y1 - y0)


def compute_displacement_correlations_along_x(
    velocity_data,
    start_frame=90,
    end_frame=110,         # starting times t in [start_frame, end_frame)
    tau_frames=5,          # lag m (in frames)
    dx_um=12.9,            # your 30px*0.43um/px
    # ROI selection (match your old choices; you can generalize to a mask later)
    y_halfwidth=4,         # mid_y +/- 4
    start_x=10,
    end_x_margin=5,
    subtract_mode="mean"   # "mean" subtraction per time
):
    """Returns (r_um, Dpar, Dper, Dpar_n, Dper_n, xi_um)."""

    # --- keys & shape
    keys = sorted(velocity_data.keys())
    # pick any available frame to get shape
    UX0, UY0 = velocity_data[keys[0]]
    Ny, Nx = UX0.shape

    # --- ROI indices
    mid_y = Ny // 2
    y0 = max(mid_y - y_halfwidth, 0)
    y1 = min(mid_y + y_halfwidth, Ny)

    x0 = start_x
    x1 = Nx - end_x_margin
    if x1 <= x0 + 2:
        raise ValueError("ROI too small in x after margins.")

    # --- check we have enough frames
    # We need frames up to (end_frame + tau_frames - 2)
    max_needed = end_frame + tau_frames - 2
    if max_needed not in velocity_data:
        # fall back: cap end_frame
        max_key = max(keys)
        end_frame = max_key - tau_frames + 2
        if end_frame <= start_frame:
            raise ValueError("Not enough frames for requested tau_frames in this file.")

    # --- load per-frame displacement increments for the needed range
    # We need frames t = start_frame .. end_frame+tau_frames-2 inclusive
    frame_list = list(range(start_frame, end_frame + tau_frames - 1))
    Ux = np.stack([velocity_data[t][0][y0:y1, x0:x1] for t in frame_list], axis=0)
    Uy = np.stack([velocity_data[t][1][y0:y1, x0:x1] for t in frame_list], axis=0)

    # --- build lagged displacement via cumulative sum
    # U_tau(t) = sum_{q=0}^{m-1} U(t+q)
    cUx = np.concatenate([np.zeros_like(Ux[:1]), np.cumsum(Ux, axis=0)], axis=0)
    cUy = np.concatenate([np.zeros_like(Uy[:1]), np.cumsum(Uy, axis=0)], axis=0)
    Ux_tau = cUx[tau_frames:] - cUx[:-tau_frames]  # shape: (end_frame-start_frame, Ny_roi, Nx_roi)
    Uy_tau = cUy[tau_frames:] - cUy[:-tau_frames]

    # --- subtract global mode per time
    if subtract_mode == "mean":
        Ux_tau = Ux_tau - Ux_tau.mean(axis=(1,2), keepdims=True)
        Uy_tau = Uy_tau - Uy_tau.mean(axis=(1,2), keepdims=True)

    T, Ny_roi, Nx_roi = Ux_tau.shape

    # --- compute all-pairs correlations along x at each separation s
    Dpar = np.zeros(Nx_roi, dtype=np.float64)
    Dper = np.zeros(Nx_roi, dtype=np.float64)

    for s in range(Nx_roi):
        # valid x pairs are [:Nx_roi-s] and [s:]
        Dpar[s] = np.mean(Ux_tau[:, :, :Nx_roi-s] * Ux_tau[:, :, s:])
        Dper[s] = np.mean(Uy_tau[:, :, :Nx_roi-s] * Uy_tau[:, :, s:])

    # normalized shapes for extracting xi
    Dpar0 = Dpar[0] if np.abs(Dpar[0]) > 1e-12 else np.nan
    Dper0 = Dper[0] if np.abs(Dper[0]) > 1e-12 else np.nan
    Dpar_n = Dpar / Dpar0 if not np.isnan(Dpar0) else np.full_like(Dpar, np.nan)
    Dper_n = Dper / Dper0 if not np.isnan(Dper0) else np.full_like(Dper, np.nan)

    r_um = np.arange(Nx_roi) * dx_um

    xi_um = first_crossing(r_um, Dpar_n, threshold=1/np.e)

    return r_um, Dpar, Dper, Dpar_n, Dper_n, xi_um


def analyze_tau_dependence(velocity_data, tau_list=[1, 3, 5, 10], **kwargs):
    """Correlation length vs lag time tau."""
    results = {}

    for tau_frames in tau_list:
        try:
            r_um, Dpar, Dper, Dpar_n, Dper_n, xi_um = compute_displacement_correlations_along_x(
                velocity_data, tau_frames=tau_frames, **kwargs
            )
            results[tau_frames] = {
                'r_um': r_um,
                'Dpar': Dpar,
                'Dper': Dper,
                'Dpar_n': Dpar_n,
                'Dper_n': Dper_n,
                'xi_um': xi_um
            }
        except Exception as e:
            print(f"Failed for tau={tau_frames}: {e}")

    return results


def plot_correlation_curves(results, title="Displacement Correlations"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Handle single result or multiple tau values
    if 'r_um' in results:  # single result
        results = {'τ': results}

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))

    for i, (tau, data) in enumerate(results.items()):
        r_um = data['r_um']
        Dpar_n = data['Dpar_n']
        Dper_n = data['Dper_n']
        xi_um = data['xi_um']

        label = f"τ={tau} frames (ξ={xi_um:.1f} µm)" if tau != 'τ' else f"ξ={xi_um:.1f} µm"

        ax1.plot(r_um, Dpar_n, 'o-', color=colors[i], label=label, markersize=4)
        ax2.plot(r_um, Dper_n, 's-', color=colors[i], label=label, markersize=4)

    # Add 1/e threshold line
    for ax in [ax1, ax2]:
        ax.axhline(1/np.e, color='r', linestyle='--', alpha=0.5, label='1/e threshold')
        ax.set_xlabel('Separation r (µm)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, None)
        ax.set_ylim(-0.1, 1.1)

    ax1.set_ylabel('D‖(r,τ) / D‖(0,τ)')
    ax1.set_title('Parallel (along bar)')

    ax2.set_ylabel('D⊥(r,τ) / D⊥(0,τ)')
    ax2.set_title('Perpendicular (transverse)')

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def process_all_timepoints(base_directory, tau_frames=5, frame_rate=0.1):
    timepoints = []
    xi_values = []
    all_data = {}

    # Find all folders with velocity_data.pkl
    folders = [d for d in os.listdir(base_directory)
               if os.path.isdir(os.path.join(base_directory, d)) and d.endswith('_finer')]

    for folder in tqdm(folders, desc="Processing folders"):
        # Extract timepoint
        parts = folder.split('_')
        if parts[0] == '5' and parts[1] == 'min':
            timepoint = 5
        else:
            timepoint = int(parts[0])

        # Load velocity data
        pkl_path = os.path.join(base_directory, folder, 'velocity_data.pkl')
        if not os.path.exists(pkl_path):
            continue

        try:
            with open(pkl_path, 'rb') as f:
                velocity_data = pickle.load(f)

            # Compute correlations
            r_um, Dpar, Dper, Dpar_n, Dper_n, xi_um = compute_displacement_correlations_along_x(
                velocity_data, tau_frames=tau_frames
            )

            timepoints.append(timepoint)
            xi_values.append(xi_um)
            all_data[timepoint] = {
                'r_um': r_um,
                'Dpar_n': Dpar_n,
                'Dper_n': Dper_n,
                'xi_um': xi_um
            }

        except Exception as e:
            print(f"Error processing {folder}: {e}")

    # Sort by timepoint
    sorted_indices = np.argsort(timepoints)
    timepoints = [timepoints[i] for i in sorted_indices]
    xi_values = [xi_values[i] for i in sorted_indices]

    return timepoints, xi_values, all_data


def plot_correlation_length_vs_time(timepoints, xi_values, tau_frames=5, frame_rate=0.1):
    fig, ax = plt.subplots(figsize=(10, 6))
    tau_seconds = tau_frames / frame_rate
    colors = plt.cm.viridis((np.array(timepoints) - min(timepoints)) /
                           (max(timepoints) - min(timepoints)))
    ax.scatter(timepoints, xi_values, c=colors, s=100, alpha=0.7, edgecolors='k', linewidth=0.5)
    ax.plot(timepoints, xi_values, 'k-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Incubation Time (minutes)')
    ax.set_ylabel(f'Correlation Length ξ (µm)')
    ax.set_title(f'Displacement Correlation Length (τ = {tau_seconds:.1f} s)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

