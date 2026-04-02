#!/usr/bin/env python3
"""Compute PIV + ξ for one dataset. Cy5 channel → optical flow → displacement correlation → ξ."""

import numpy as np, cv2, pickle, sys, os, time
from glob import glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'sandbox'))
from displacement_correlation import compute_displacement_correlations_along_x

def normalize_to_uint8(img, lo=1, hi=99):
    """Percentile-clip normalize to uint8 — handles hot pixels in 16-bit TIFFs."""
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    vlo, vhi = np.percentile(img_f, [lo, hi])
    img_f = np.clip((img_f - vlo) / (vhi - vlo + 1e-6), 0, 1)
    return (img_f * 255).astype(np.uint8)


def perform_PIV(image_folder, output_folder, gridSize=30, sigma=30, velScale=1.6):
    """Farneback optical flow on Cy5 frames — matches piv_contraction_analysis.py exactly."""
    files = sorted(glob(os.path.join(image_folder, '*Cy5*.tif')))
    if not files:
        return None

    from PIL import Image as PILImage
    im1 = normalize_to_uint8(np.array(PILImage.open(files[0])))
    yrange, xrange = im1.shape[:2]

    x = np.arange(gridSize/2, xrange - gridSize/2 + 1, gridSize)
    y = np.arange(gridSize/2, yrange - gridSize/2 + 1, gridSize)
    X1, Y1 = np.meshgrid(x, y)

    os.makedirs(output_folder, exist_ok=True)
    velocity_data = {}

    for tt in range(len(files) - 1):
        im2 = normalize_to_uint8(np.array(PILImage.open(files[tt + 1])))

        flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        VX = cv2.GaussianBlur(flow[..., 0], (0, 0), sigma)
        VY = cv2.GaussianBlur(flow[..., 1], (0, 0), sigma)
        UX = cv2.resize(VX, (X1.shape[1], X1.shape[0])) * velScale
        UY = cv2.resize(VY, (X1.shape[1], X1.shape[0])) * velScale
        velocity_data[tt] = (UX.copy(), UY.copy())
        im1 = im2

    with open(os.path.join(output_folder, 'velocity_data.pkl'), 'wb') as f:
        pickle.dump(velocity_data, f)

    return velocity_data


def main():
    name = sys.argv[1]
    image_folder = sys.argv[2]
    piv_output = sys.argv[3]

    t0 = time.time()

    # Step 1: PIV
    print(f'[{name}] Running PIV...', flush=True)
    vd = perform_PIV(image_folder, piv_output)
    if vd is None or len(vd) == 0:
        print(f'[{name}] PIV failed!', flush=True)
        return
    t_piv = time.time() - t0
    print(f'[{name}] PIV done: {len(vd)} frames, {t_piv:.0f}s', flush=True)

    # Step 2: ξ extraction at multiple τ
    n = len(vd)
    taus = [1, 3, 5, 10, 15]
    print(f'[{name}] Computing ξ...', flush=True)
    results = []
    for tau in taus:
        try:
            r_um, Dpar, Dper, Dpar_n, Dper_n, xi = compute_displacement_correlations_along_x(
                vd, tau_frames=tau,
                start_frame=min(90, n - 20),
                end_frame=min(110, n - 5))
            results.append((tau, tau * 10, xi))
        except Exception as e:
            results.append((tau, tau * 10, np.nan))

    elapsed = time.time() - t0
    xi_strs = ', '.join([f'τ={r[1]}s: ξ={r[2]:.1f}µm' for r in results if not np.isnan(r[2])])
    print(f'[{name}] DONE ({elapsed:.0f}s): {xi_strs}', flush=True)


if __name__ == '__main__':
    main()
