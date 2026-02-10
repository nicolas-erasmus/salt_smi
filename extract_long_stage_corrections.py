import os
import csv
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.centroids import centroid_2dg
from scipy.optimize import curve_fit

# ----------------------------
# Configuration
# ----------------------------
fits_dir = os.getcwd() + "/300_ver_1/cam3_images/"
corner_size = 200
px_to_um =  1.07 # from images 280px = 300um
output_csv = os.getcwd() + "/300_ver_1/long_stage_corrections.csv"

# ----------------------------
# Helper functions
# ----------------------------
def corner_means(image, size=200):
    ny, nx = image.shape
    corners = [
        image[0:size, 0:size],
        image[0:size, nx-size:nx],
        image[ny-size:ny, 0:size],
        image[ny-size:ny, nx-size:nx]
    ]
    return [np.nanmean(c) for c in corners]


def max_radius_to_edge(x0, y0, nx, ny):
    return int(min(x0, y0, nx - x0 - 1, ny - y0 - 1))


def find_centroid_2dg(image):
    mask = ~np.isfinite(image)
    return centroid_2dg(image, mask=mask)



results = []
fits_files = sorted(f for f in os.listdir(fits_dir) if f.lower().endswith(".fits"))

for fname in fits_files:
    fiber_path = os.path.join(fits_dir, fname)

    with fits.open(fiber_path) as hdul:
        fiber_data = hdul[0].data.astype(float)


    fiber_sub = fiber_data - np.mean(corner_means(fiber_data, corner_size))

    x_fib, y_fib = find_centroid_2dg(fiber_sub)
    offset = (fiber_sub.shape[0]/2 - y_fib)*px_to_um
   

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)

    im = ax1.imshow(
        fiber_sub,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(fiber_sub, 5),
        vmax=np.percentile(fiber_sub, 99)
    )
    ax1.plot(x_fib, y_fib, "+", color="red", markersize=18, mew=2)
    ax1.set_title(f"{fname}\nFiber Image")
    ax1.set_xlabel("X [pixels]")
    ax1.set_ylabel("Y [pixels]")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # plt.show()
    
    fiber_number = fname.split("_")[0]
    results.append((fiber_number, offset))
    
    plt.savefig(fits_dir+f"/{fname.split(".")[0]}.pdf", bbox_inches='tight', dpi=150)
    
# ----------------------------
# Write CSV
# ----------------------------
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fiber_number", "Offset"])
    writer.writerows(results)

print(f"Saved results to {output_csv}")
