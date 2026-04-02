import numpy as np, tifffile, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os, warnings
warnings.filterwarnings("ignore")

base = '/Users/shichenliu/Documents/0529_22/sequences'
cy, cx, r = 960, 1024, 775
pixel_size_um = 0.43

reps = [
    ('Disconnected',       'disconnected_1', 13.4),
    ('Connected, floppy',  'not_rigid_1',    13.4),
    ('Marginally rigid',   'barely_rigid_2', 18.8),
    ('Rigid',              'rigid_1',        18.8),
]

target_times_s = [None, 10 * 13.4, 20 * 13.4, 40 * 13.4]
col_titles = ['0 min', '2.2 min', '4.5 min', '8.9 min']
line_color = '#E8555A'

frame1_means = []
for label, seq_name, dt in reps:
    files = sorted([f for f in os.listdir(os.path.join(base, seq_name)) if f.endswith('.tif')])
    img = tifffile.imread(os.path.join(base, seq_name, files[1]))
    frame1_means.append(img[cy-r:cy+r, cx-r:cx+r].astype(np.float64).mean())
ref_mean = min(frame1_means)
scale_factors = [ref_mean / m for m in frame1_means]

all_data = []
for idx, (label, seq_name, dt) in enumerate(reps):
    seq_dir = os.path.join(base, seq_name)
    files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.tif')])
    T = len(files)
    sf = scale_factors[idx]
    img_ref = tifffile.imread(os.path.join(seq_dir, files[1]))
    roi_ref = img_ref[cy-r:cy+r, cx-r:cx+r].astype(np.float64) * sf
    flat = gaussian_filter(roi_ref, sigma=50)
    flat[flat < 1] = 1
    flat_norm = flat / flat.mean()
    frames = []
    for t_target in target_times_s:
        fi = 1 if t_target is None else min(int(round(t_target / dt)), T - 1)
        img = tifffile.imread(os.path.join(seq_dir, files[fi]))
        roi = img[cy-r:cy+r, cx-r:cx+r].astype(np.float64) * sf
        roi_corr = roi / flat_norm
        mid_y = roi_corr.shape[0] // 2
        band = roi_corr[mid_y-10:mid_y+10, :].mean(axis=0)
        v1, v2 = np.percentile(roi_corr, 1), np.percentile(roi_corr, 99.5)
        disp = np.clip((roi_corr - v1) / (v2 - v1 + 1e-10), 0, 1)
        frames.append({'disp': disp, 'band': band, 'mid_y': mid_y})
    all_data.append({'label': label, 'frames': frames})

global_min = min(fd['band'].min() for d in all_data for fd in d['frames'])
global_max = max(fd['band'].max() for d in all_data for fd in d['frames'])

n_cond = len(reps)
n_times = len(target_times_s)
n_rows = n_cond * 2

fig, axes = plt.subplots(n_rows, n_times, figsize=(5*n_times, 3.5*n_rows),
                          gridspec_kw={'height_ratios': [1.2, 0.8]*n_cond})

for i, d in enumerate(all_data):
    row_img = i * 2
    row_line = i * 2 + 1
    for j, fd in enumerate(d['frames']):
        ax_img = axes[row_img, j]
        ax_img.imshow(fd['disp'], cmap='gray_r')
        ax_img.axhline(fd['mid_y'], color=line_color, lw=1.5, alpha=0.7)
        ax_img.axis('off')
        if i == 0:
            ax_img.set_title(col_titles[j], fontsize=20, fontweight='bold')
        if j == 0:
            ax_img.text(-0.05, 0.5, d['label'], transform=ax_img.transAxes,
                       fontsize=18, fontweight='bold', color='black',
                       va='center', ha='right', rotation=90)

        ax_line = axes[row_line, j]
        x_um = np.arange(len(fd['band'])) * pixel_size_um
        band_norm = (fd['band'] - global_min) / (global_max - global_min)
        band_norm = np.maximum(band_norm, 1e-3)
        ax_line.plot(x_um, band_norm, '-', color=line_color, lw=2)
        ax_line.set_xlim(0, x_um[-1])
        ax_line.set_yscale('log')
        ax_line.set_ylim(1e-3, 1.5)
        ax_line.grid(True, alpha=0.2)
        ax_line.tick_params(labelsize=12)
        if i == n_cond - 1:
            ax_line.set_xlabel('Position (µm)', fontsize=16)
        else:
            ax_line.set_xticklabels([])

plt.tight_layout()
fig.savefig(os.path.join(base, 'analysis', 'combined_4reps.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved combined_4reps.png")
