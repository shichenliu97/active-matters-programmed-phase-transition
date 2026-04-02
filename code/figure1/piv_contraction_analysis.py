"""
PIV Contraction Speed Analysis
==============================
Extract and compare aster contraction speeds from Cy5 channel PIV data.

Compares:
- 1120 Stokes data (local aster contraction with beads)
- Figure 1 Correlation data (global network contraction)

Usage:
    python piv_contraction_analysis.py --dataset 1120
    python piv_contraction_analysis.py --compare

Author: Generated 2026-01-25
"""

import numpy as np
import cv2
from PIL import Image
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path
from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================

# PIV parameters (matching original PIV.ipynb)
GRID_SIZE = 30
SIGMA = 30
VEL_SCALE = 1.6
SLICE_HEIGHT = 8  # central slice for speed calculation

# Physical parameters
PIXEL_SIZE = 0.43  # um/pixel
DT_S = 10  # seconds between frames
SCALE = PIXEL_SIZE / DT_S  # um/s per px/frame

# Data paths
BASE_PATH_1120 = Path("/media/shichenliu/HHD_Storage_1013/Dropbox/Academics/PhD_phase/Thomson_Lab/Active_matter/local_to_global_pre-print/data/figure_2/stokes/1120")
BASE_PATH_FIG1 = Path("/media/shichenliu/HHD_Storage_1013/Dropbox/Academics/PhD_phase/Thomson_Lab/Active_matter/local_to_global_pre-print/data/figure_1/0104PIV_data_0104")
CONTRACTION_CSV = Path("/home/shichenliu/Documents/git/active-matters-programmed-phase-transition/analysis/sandbox/contraction_speed.csv")

# Dataset mapping: (incubation_time, position) -> folder path
DATASETS_1120 = {
    (5, 0): 'mid_pH_test_5min_1/Pos0',
    (5, 1): 'mid_pH_test_5min_1/Pos1',
    (5, 2): 'mid_pH_test_5min_1/Pos2',
    (60, 0): 'mid_pH_test_60min_1/Pos0',
    (60, 1): 'mid_pH_test_60min_1/Pos1',
    (60, 2): 'mid_pH_test_60min_1/Pos2',
    (95, 0): 'mid_pH_test_95min_1/Pos0',
    (95, 1): 'mid_pH_test_95min_1/Pos1',
    (95, 2): 'mid_pH_test_95min_1/Pos2',
    (125, 0): 'mid_pH_test_125min_1/Pos0',
    (125, 1): 'mid_pH_test_125min_1/Pos1',
    (125, 2): 'mid_pH_test_125min_1/Pos2',
}

# Position index to CSV suffix mapping
POSITION_TO_CSV = {0: '_1', 1: '_2', 2: '_3'}


# ============================================================
# PIV FUNCTIONS
# ============================================================

def run_piv_on_tiffs(image_folder, channel='Cy5', n_frames=None, verbose=True, return_velocity_data=False):
    """
    Run Farneback optical flow PIV on TIFF image sequence.

    Args:
        image_folder: path to folder with TIFF images
        channel: channel name to filter (e.g., 'Cy5', 'Brightfield')
        n_frames: limit number of frames (None for all)
        verbose: print progress
        return_velocity_data: if True, also return velocity_data dict for pkl

    Returns:
        DataFrame with columns: frame, time_min, mean_speed_um_s, etc.
        If return_velocity_data=True, returns (DataFrame, velocity_data_dict)
    """
    tiff_files = sorted(glob(os.path.join(image_folder, f'*{channel}*.tif')))

    if n_frames:
        tiff_files = tiff_files[:n_frames]

    if len(tiff_files) < 2:
        return (None, None) if return_velocity_data else None

    if verbose:
        print(f"  Processing {len(tiff_files)} {channel} frames...")

    # Load and normalize first image
    im1 = np.array(Image.open(tiff_files[0]))
    im1 = normalize_to_uint8(im1)

    yrange, xrange = im1.shape[:2]

    # Define grid
    x = np.arange(GRID_SIZE/2, xrange - GRID_SIZE/2 + 1, GRID_SIZE)
    y = np.arange(GRID_SIZE/2, yrange - GRID_SIZE/2 + 1, GRID_SIZE)
    X1, Y1 = np.meshgrid(x, y)

    ny = X1.shape[0]
    y0, y1 = ny // 2 - SLICE_HEIGHT // 2, ny // 2 + SLICE_HEIGHT // 2

    results = []
    velocity_data = {} if return_velocity_data else None

    for i in range(len(tiff_files) - 1):
        im2 = np.array(Image.open(tiff_files[i + 1]))
        im2 = normalize_to_uint8(im2)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Extract and smooth velocity components
        VX = cv2.GaussianBlur(flow[..., 0], (0, 0), SIGMA)
        VY = cv2.GaussianBlur(flow[..., 1], (0, 0), SIGMA)

        # Resize to grid and apply velocity scale
        UX = cv2.resize(VX, (X1.shape[1], X1.shape[0])) * VEL_SCALE
        UY = cv2.resize(VY, (X1.shape[1], X1.shape[0])) * VEL_SCALE

        # Store velocity data if requested
        if return_velocity_data:
            velocity_data[i] = (UX.copy(), UY.copy())

        # Compute speed in central slice
        ux_slice = UX[y0:y1, :]
        uy_slice = UY[y0:y1, :]
        speed = np.sqrt(ux_slice**2 + uy_slice**2)

        results.append({
            'frame': i,
            'time_s': i * DT_S,
            'time_min': i * DT_S / 60,
            'mean_vx_um_s': np.mean(ux_slice) * SCALE,
            'mean_vy_um_s': np.mean(uy_slice) * SCALE,
            'mean_speed_um_s': np.mean(speed) * SCALE,
            'rms_speed_um_s': np.sqrt(np.mean(speed**2)) * SCALE,
            'max_speed_um_s': np.max(speed) * SCALE,
        })

        im1 = im2

        if verbose and (i + 1) % 50 == 0:
            print(f"    Frame {i + 1}/{len(tiff_files) - 1}")

    df = pd.DataFrame(results)
    if return_velocity_data:
        return df, velocity_data
    return df


def normalize_to_uint8(img):
    """Normalize image to uint8 range."""
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    img_norm = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-6)
    return (img_norm * 255).astype(np.uint8)


def run_piv_sampled(image_folder, frame_indices, channel='Cy5'):
    """
    Run PIV on specific frames only (faster for comparison).

    Args:
        image_folder: path to folder with TIFF images
        frame_indices: list of frame indices to process
        channel: channel name

    Returns:
        DataFrame with speed at sampled frames
    """
    tiff_files = sorted(glob(os.path.join(image_folder, f'*{channel}*.tif')))

    results = []

    for start_idx in frame_indices:
        if start_idx + 1 >= len(tiff_files):
            continue

        im1 = normalize_to_uint8(np.array(Image.open(tiff_files[start_idx])))
        im2 = normalize_to_uint8(np.array(Image.open(tiff_files[start_idx + 1])))

        flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        VX = cv2.GaussianBlur(flow[..., 0], (0, 0), SIGMA) * VEL_SCALE
        VY = cv2.GaussianBlur(flow[..., 1], (0, 0), SIGMA) * VEL_SCALE

        # Resize
        xrange, yrange = im1.shape[1], im1.shape[0]
        x = np.arange(GRID_SIZE/2, xrange - GRID_SIZE/2 + 1, GRID_SIZE)
        y = np.arange(GRID_SIZE/2, yrange - GRID_SIZE/2 + 1, GRID_SIZE)
        X1, Y1 = np.meshgrid(x, y)

        UX = cv2.resize(VX, (X1.shape[1], X1.shape[0]))
        UY = cv2.resize(VY, (X1.shape[1], X1.shape[0]))

        ny = UX.shape[0]
        y0, y1 = ny // 2 - SLICE_HEIGHT // 2, ny // 2 + SLICE_HEIGHT // 2
        speed = np.sqrt(UX[y0:y1, :]**2 + UY[y0:y1, :]**2)

        results.append({
            'frame': start_idx,
            'time_min': start_idx * DT_S / 60,
            'mean_speed_um_s': np.mean(speed) * SCALE
        })

    return pd.DataFrame(results)


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def process_1120_datasets(n_frames=None, sampled=False, positions=None, save_pkl=False, output_dir=None):
    """Process all 1120 stokes datasets.

    Args:
        n_frames: limit number of frames (None for all)
        sampled: use sampled frames only (faster)
        positions: list of positions to process (default [0,1,2] for all)
        save_pkl: if True, save velocity_data.pkl files
        output_dir: directory to save pkl files (required if save_pkl=True)
    """
    all_results = []
    frame_indices = [0, 5, 10, 20, 50, 100, 150] if sampled else None

    if positions is None:
        positions = [0, 1, 2]

    if save_pkl and output_dir is None:
        output_dir = Path('/tmp/1120_piv')

    if save_pkl:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for (inc_time, position), folder_path in DATASETS_1120.items():
        if position not in positions:
            continue

        img_folder = BASE_PATH_1120 / folder_path

        if not img_folder.exists():
            print(f"  Skipping {folder_path}: not found")
            continue

        print(f"\nProcessing {inc_time} min, Pos{position} ({folder_path})...")

        if sampled:
            df = run_piv_sampled(str(img_folder), frame_indices)
            velocity_data = None
        else:
            if save_pkl:
                df, velocity_data = run_piv_on_tiffs(str(img_folder), n_frames=n_frames, return_velocity_data=True)
            else:
                df = run_piv_on_tiffs(str(img_folder), n_frames=n_frames)
                velocity_data = None

        if df is not None:
            df['incubation_time'] = inc_time
            df['position'] = position
            df['source'] = '1120_stokes'
            all_results.append(df)

            # Save velocity_data.pkl if requested
            if save_pkl and velocity_data is not None:
                pkl_folder = output_dir / f"{inc_time}min_Pos{position}"
                pkl_folder.mkdir(parents=True, exist_ok=True)
                pkl_path = pkl_folder / 'velocity_data.pkl'
                with open(pkl_path, 'wb') as f:
                    pickle.dump(velocity_data, f)
                print(f"  Saved: {pkl_path}")

    return pd.concat(all_results, ignore_index=True) if all_results else None


def load_figure1_data():
    """Load pre-computed Figure 1 correlation PIV data."""
    if CONTRACTION_CSV.exists():
        df = pd.read_csv(CONTRACTION_CSV)
        df['source'] = 'figure_1_correlation'
        return df
    return None


def compare_datasets(df_1120, df_fig1, output_dir=None):
    """
    Compare contraction speeds between 1120 and Figure 1 data.

    Returns summary DataFrame and generates plots.
    """
    if output_dir is None:
        output_dir = Path('/tmp')

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Summary by incubation time
    print("\n" + "=" * 60)
    print("CONTRACTION SPEED COMPARISON")
    print("=" * 60)

    comparison_data = []

    for inc_time in [5, 35, 65, 125]:
        d1 = df_1120[df_1120['incubation_time'] == inc_time]

        # Use 120 min for Figure 1 comparison with 125 min
        fig1_time = 120 if inc_time == 125 else inc_time
        d2 = df_fig1[df_fig1['incubation_time'] == fig1_time]

        if len(d1) > 0 and len(d2) > 0:
            # Early phase (frames 5-15)
            early_1120 = d1[(d1['frame'] >= 5) & (d1['frame'] <= 15)]['mean_speed_um_s'].mean()
            early_fig1 = d2[(d2['frame'] >= 5) & (d2['frame'] <= 15)]['mean_speed_um_s'].mean()

            # Late phase (frames 50-100)
            late_1120 = d1[(d1['frame'] >= 50) & (d1['frame'] <= 100)]['mean_speed_um_s'].mean()
            late_fig1 = d2[(d2['frame'] >= 50) & (d2['frame'] <= 100)]['mean_speed_um_s'].mean()

            comparison_data.append({
                'incubation_time': inc_time,
                'early_1120': early_1120,
                'early_fig1': early_fig1,
                'late_1120': late_1120,
                'late_fig1': late_fig1,
                'early_ratio': early_fig1 / early_1120 if early_1120 > 0 else np.nan,
                'late_ratio': late_fig1 / late_1120 if late_1120 > 0 else np.nan,
            })

            print(f"\n{inc_time} min:")
            print(f"  Early (frames 5-15):  1120={early_1120:.4f}, Fig1={early_fig1:.4f} µm/s")
            print(f"  Late (frames 50-100): 1120={late_1120:.4f}, Fig1={late_fig1:.4f} µm/s")

    comparison_df = pd.DataFrame(comparison_data)

    # Generate comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, inc_time in enumerate([5, 35, 65, 125]):
        ax = axes.flatten()[idx]

        # 1120 data
        d1 = df_1120[df_1120['incubation_time'] == inc_time]
        if len(d1) > 0:
            ax.plot(d1['time_min'], d1['mean_speed_um_s'], 'b-', lw=2,
                   label='1120 Stokes (local)')

        # Figure 1 data
        fig1_time = 120 if inc_time == 125 else inc_time
        d2 = df_fig1[df_fig1['incubation_time'] == fig1_time]
        if len(d2) > 0:
            for folder, grp in d2.groupby('folder'):
                ax.plot(grp['time_min'], grp['mean_speed_um_s'], 'r-', alpha=0.3, lw=1)
            ax.plot([], [], 'r-', label='Fig1 Correlation (global)')

        ax.set_xlabel('Time in movie (min)')
        ax.set_ylabel('Speed (µm/s)')
        ax.set_title(f'{inc_time} min incubation')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 30)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'contraction_speed_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'contraction_speed_comparison.png'}")

    return comparison_df


# ============================================================
# BEAD VS PIV COMPARISON
# ============================================================

def extract_piv_at_positions(image_folder, positions_df, channel='Cy5'):
    """
    Compute PIV and extract velocity at specified (x, y, frame) positions.

    Args:
        image_folder: path to folder with TIFF images
        positions_df: DataFrame with columns [x, y, frame]
        channel: channel name

    Returns:
        DataFrame with [frame, x, y, piv_vx_um_s, piv_vy_um_s, piv_speed_um_s]
    """
    tiff_files = sorted(glob(os.path.join(image_folder, f'*{channel}*.tif')))
    if len(tiff_files) < 2:
        return None

    frames_needed = sorted(positions_df['frame'].unique())
    max_frame = max(frames_needed)

    if max_frame >= len(tiff_files) - 1:
        max_frame = len(tiff_files) - 2

    results = []
    im1 = normalize_to_uint8(np.array(Image.open(tiff_files[0])))

    for frame_idx in range(max_frame + 1):
        im2 = normalize_to_uint8(np.array(Image.open(tiff_files[frame_idx + 1])))

        if frame_idx in frames_needed:
            flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            VX = cv2.GaussianBlur(flow[..., 0], (0, 0), SIGMA) * VEL_SCALE
            VY = cv2.GaussianBlur(flow[..., 1], (0, 0), SIGMA) * VEL_SCALE

            frame_positions = positions_df[positions_df['frame'] == frame_idx]
            for _, row in frame_positions.iterrows():
                x, y = int(row['x']), int(row['y'])
                if 0 <= y < VX.shape[0] and 0 <= x < VX.shape[1]:
                    results.append({
                        'frame': frame_idx,
                        'x': row['x'],
                        'y': row['y'],
                        'piv_vx_um_s': VX[y, x] * SCALE,
                        'piv_vy_um_s': VY[y, x] * SCALE,
                        'piv_speed_um_s': np.sqrt(VX[y, x]**2 + VY[y, x]**2) * SCALE
                    })

        im1 = im2

    return pd.DataFrame(results)


def compare_bead_vs_piv(bead_tracks_csv, image_folder, channel='Cy5'):
    """
    Compare bead-derived velocities with PIV velocities at bead locations.

    Args:
        bead_tracks_csv: path to CSV with bead tracks (from BeadTracker)
        image_folder: path to image folder for PIV

    Returns:
        DataFrame with bead and PIV velocities merged
    """
    bead_df = pd.read_csv(bead_tracks_csv)

    positions_df = bead_df[['x', 'y', 'frame', 'particle']].copy()
    positions_df = positions_df.dropna()

    print(f"Extracting PIV at {len(positions_df)} bead positions...")
    piv_df = extract_piv_at_positions(image_folder, positions_df, channel)

    if piv_df is None or len(piv_df) == 0:
        return None

    bead_subset = bead_df[['x', 'y', 'frame', 'particle', 'dx', 'dy',
                           'velocity_um_per_s']].copy()
    bead_subset['bead_vx_um_s'] = bead_subset['dx'] * PIXEL_SIZE / DT_S
    bead_subset['bead_vy_um_s'] = bead_subset['dy'] * PIXEL_SIZE / DT_S

    merged = pd.merge(bead_subset, piv_df, on=['x', 'y', 'frame'], how='inner')

    return merged


def plot_bead_piv_comparison(comparison_df, output_path):
    """
    Generate scatter plot comparing bead vs PIV velocities.

    Args:
        comparison_df: DataFrame from compare_bead_vs_piv
        output_path: path for output figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    bead_speed = comparison_df['velocity_um_per_s'].dropna()
    piv_speed = comparison_df['piv_speed_um_s'].dropna()

    valid = bead_speed.notna() & piv_speed.notna()
    bead_speed = bead_speed[valid].values
    piv_speed = piv_speed[valid].values

    # Speed comparison
    ax = axes[0]
    ax.scatter(bead_speed, piv_speed, alpha=0.3, s=10)
    max_val = max(bead_speed.max(), piv_speed.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
    ax.set_xlabel('Bead velocity (µm/s)')
    ax.set_ylabel('PIV velocity (µm/s)')
    ax.set_title('Speed comparison')

    slope, intercept, r_value, _, _ = stats.linregress(bead_speed, piv_speed)
    ax.text(0.05, 0.95, f'R² = {r_value**2:.3f}\nslope = {slope:.2f}',
            transform=ax.transAxes, va='top', fontsize=10)
    ax.legend()

    # Vx comparison
    ax = axes[1]
    bead_vx = comparison_df['bead_vx_um_s'].dropna().values
    piv_vx = comparison_df['piv_vx_um_s'].dropna().values
    min_len = min(len(bead_vx), len(piv_vx))
    ax.scatter(bead_vx[:min_len], piv_vx[:min_len], alpha=0.3, s=10)
    ax.set_xlabel('Bead Vx (µm/s)')
    ax.set_ylabel('PIV Vx (µm/s)')
    ax.set_title('Vx comparison')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    # Vy comparison
    ax = axes[2]
    bead_vy = comparison_df['bead_vy_um_s'].dropna().values
    piv_vy = comparison_df['piv_vy_um_s'].dropna().values
    min_len = min(len(bead_vy), len(piv_vy))
    ax.scatter(bead_vy[:min_len], piv_vy[:min_len], alpha=0.3, s=10)
    ax.set_xlabel('Bead Vy (µm/s)')
    ax.set_ylabel('PIV Vy (µm/s)')
    ax.set_title('Vy comparison')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    rmse = np.sqrt(np.mean((bead_speed - piv_speed)**2))
    print(f"  R² = {r_value**2:.3f}, slope = {slope:.2f}, RMSE = {rmse:.4f} µm/s")

    return {'r2': r_value**2, 'slope': slope, 'rmse': rmse}


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='PIV Contraction Speed Analysis')
    parser.add_argument('--dataset', type=str, choices=['1120', 'fig1', 'both'],
                       default='both', help='Which dataset to process')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison between datasets')
    parser.add_argument('--n_frames', type=int, default=None,
                       help='Limit number of frames to process')
    parser.add_argument('--sampled', action='store_true',
                       help='Use sampled frames only (faster)')
    parser.add_argument('--output_dir', type=str, default='/tmp',
                       help='Output directory for figures')
    parser.add_argument('--positions', type=int, nargs='+', default=None,
                       help='Positions to process (default: all [0,1,2])')
    parser.add_argument('--save_pkl', action='store_true',
                       help='Save velocity_data.pkl files for correlation analysis')
    parser.add_argument('--bead_comparison', action='store_true',
                       help='Run bead vs PIV velocity comparison')
    parser.add_argument('--bead_tracks', type=str, default=None,
                       help='Path to bead tracks CSV for comparison')
    parser.add_argument('--image_folder', type=str, default=None,
                       help='Image folder for bead comparison')

    args = parser.parse_args()

    df_1120 = None
    df_fig1 = None

    if args.bead_comparison:
        if args.bead_tracks is None or args.image_folder is None:
            print("Error: --bead_comparison requires --bead_tracks and --image_folder")
            return

        print(f"Running bead vs PIV comparison...")
        comparison_df = compare_bead_vs_piv(args.bead_tracks, args.image_folder)
        if comparison_df is not None:
            output_csv = Path(args.output_dir) / 'bead_vs_piv_comparison.csv'
            comparison_df.to_csv(output_csv, index=False)
            print(f"Saved: {output_csv}")

            output_fig = Path(args.output_dir) / 'bead_vs_piv_scatter.png'
            plot_bead_piv_comparison(comparison_df, output_fig)
        return

    if args.dataset in ['1120', 'both']:
        print("Processing 1120 Stokes data...")
        pkl_output = Path(args.output_dir) / '1120_piv' if args.save_pkl else None
        df_1120 = process_1120_datasets(n_frames=args.n_frames, sampled=args.sampled,
                                        positions=args.positions, save_pkl=args.save_pkl,
                                        output_dir=pkl_output)
        if df_1120 is not None:
            df_1120.to_csv(Path(args.output_dir) / '1120_contraction_speed.csv', index=False)
            print(f"Saved: {args.output_dir}/1120_contraction_speed.csv")

    if args.dataset in ['fig1', 'both']:
        print("\nLoading Figure 1 data...")
        df_fig1 = load_figure1_data()

    if args.compare and df_1120 is not None and df_fig1 is not None:
        compare_datasets(df_1120, df_fig1, args.output_dir)


if __name__ == '__main__':
    main()
