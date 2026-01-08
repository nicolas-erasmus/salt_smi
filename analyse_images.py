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
fits_dir = os.getcwd() + "/300_ver_1/cam4_images/"
reference_fits = os.getcwd() + "/300_ver_1/reference_image/direct_reference1.fits"
corner_size = 200
fratio_ref = 4.2
output_csv = "fiber_frd_ee90_results.csv"

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


def cumulative_radial_profile(image, x0, y0):
    ny, nx = image.shape
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    r_int = r.astype(int)

    max_r = max_radius_to_edge(x0, y0, nx, ny)
    cumulative_counts = np.zeros(max_r + 1)

    for rad in range(max_r + 1):
        cumulative_counts[rad] = np.nansum(image[r_int <= rad])

    return np.arange(max_r + 1), cumulative_counts


def find_centroid_2dg(image):
    mask = ~np.isfinite(image)
    return centroid_2dg(image, mask=mask)


def quad_model(r, A):
    return A * r**2


def add_fratio_axis(ax, r_cross, f_ref):
    def px_to_f(px):
        return (px / r_cross) * f_ref

    def f_to_px(f):
        return (f / f_ref) * r_cross

    secax = ax.secondary_xaxis("top", functions=(px_to_f, f_to_px))
    secax.set_xlabel("f-ratio")

# ----------------------------
# Reference image (cam4) analysis
# ----------------------------
with fits.open(reference_fits) as hdul:
    ref_data = hdul[0].data.astype(float)

ref_bg = np.mean(corner_means(ref_data, corner_size))
ref_sub = ref_data - ref_bg

x_ref, y_ref = find_centroid_2dg(ref_sub)
r_ref, c_ref = cumulative_radial_profile(ref_sub, x_ref, y_ref)
c_ref_norm = c_ref / np.nanmax(c_ref)

r75 = np.interp(0.75, c_ref_norm, r_ref)
fit_mask = c_ref_norm <= 0.75

popt, _ = curve_fit(
    quad_model,
    r_ref[fit_mask],
    c_ref_norm[fit_mask],
    p0=(1.0 / r75**2)
)

A_fit = popt[0]
r_cross = np.sqrt(1.0 / A_fit)

# ----------------------------
# Reference plot
# ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

im = ax1.imshow(
    ref_sub,
    origin="lower",
    cmap="gray",
    vmin=np.percentile(ref_sub, 5),
    vmax=np.percentile(ref_sub, 99)
)
ax1.plot(x_ref, y_ref, "+", color="red", markersize=18, mew=2)
ax1.set_title("Reference Image\n2D Gaussian Centroid")
ax1.set_xlabel("X [pixels]")
ax1.set_ylabel("Y [pixels]")
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

ax2.plot(r_ref, c_ref_norm, "k", lw=2, label="Reference")
ax2.plot(r_ref, quad_model(r_ref, A_fit), "r--", lw=2, label=r"$A r^2$")
ax2.axvline(r_cross, color="blue", ls=":", lw=2,
            label=f"100% @ {r_cross:.1f}px")
ax2.set_xlabel("Radius [pixels]")
ax2.set_ylabel("Normalised cumulative counts")
ax2.set_ylim(0, 1.05)
ax2.set_title("Reference Cumulative Profile")
ax2.legend(fontsize=8)
add_fratio_axis(ax2, r_cross, fratio_ref)

plt.show()

# ----------------------------
# Main loop
# ----------------------------
results = []

fits_files = sorted(f for f in os.listdir(fits_dir) if f.lower().endswith(".fits"))

for fname in fits_files:
    fiber_path = os.path.join(fits_dir, fname)
    ref_flux_path = os.path.join(
        fits_dir.replace("cam4", "cam2"),
        fname.replace("cam4", "cam2")
    )

    with fits.open(fiber_path) as hdul:
        fiber_exp_time = hdul[0].header["EXPTIME"]
        fiber_gain = hdul[0].header["EGAINSAV"]
        fiber_data = (hdul[0].data.astype(float)/fiber_exp_time)*fiber_gain

    with fits.open(ref_flux_path) as hdul:
        ref_flux_exp_time = hdul[0].header["EXPTIME"]
        ref_flux_gain = hdul[0].header["EGAINSAV"]
        ref_flux_data = (hdul[0].data.astype(float)/ref_flux_exp_time)*ref_flux_gain

    fiber_sub = fiber_data - np.mean(corner_means(fiber_data, corner_size))
    ref_flux_sub = ref_flux_data - np.mean(corner_means(ref_flux_data, corner_size))

    x_fib, y_fib = find_centroid_2dg(fiber_sub)
    x_flux_ref, y_flux_ref = find_centroid_2dg(ref_flux_sub)

    r_fib, c_fib = cumulative_radial_profile(fiber_sub, x_fib, y_fib)
    r_flux_ref, c_flux_ref = cumulative_radial_profile(ref_flux_sub, x_flux_ref, y_flux_ref)

    c_fib_norm = c_fib / np.nanmax(c_fib)
    c_flux_ref_norm = c_flux_ref / np.nanmax(c_flux_ref)

    r90_fib = np.interp(0.90, c_fib_norm, r_fib)
    r90_flux_ref = np.interp(0.90, c_flux_ref_norm, r_flux_ref)

    C90_fib = np.interp(r90_fib, r_fib, c_fib)
    C90_flux_ref = np.interp(r90_flux_ref, r_flux_ref, c_flux_ref)

    ee90_ratio = C90_fib / C90_flux_ref

    frac_at_rcross = np.interp(r_cross, r_fib, c_fib_norm)
    frd = 1.0 - frac_at_rcross

    fiber_number = fname.split("_")[0]
    results.append((fiber_number, frd, (1-ee90_ratio)))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5), constrained_layout=True)

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

    ax2.plot(r_fib, c_fib_norm, color="gray", lw=2, label="Fiber")
    ax2.plot(r_ref, c_ref_norm, "k", lw=2, label="Reference")
    ax2.axvline(r_cross, color="blue", ls=":", lw=2)
    ax2.axhline(frac_at_rcross, color="green", ls=":", lw=2,
                label=f"{frac_at_rcross*100:.1f}% @ r_cross")
    ax2.set_xlabel("Radius [pixels]")
    ax2.set_ylabel("Normalised cumulative counts")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("FRD Diagnostic")
    ax2.legend(fontsize=8)
    add_fratio_axis(ax2, r_cross, fratio_ref)
    

    im = ax3.imshow(
        ref_flux_sub,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(ref_flux_sub, 5),
        vmax=np.percentile(ref_flux_sub, 99)
    )
    ax3.plot(x_flux_ref, y_flux_ref, "+", color="red", markersize=18, mew=2)
    ax3.set_title(f"{fname.replace("cam4", "cam2")}\nRef Flux Image")
    ax3.set_xlabel("X [pixels]")
    ax3.set_ylabel("Y [pixels]")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    ax4.plot(r_fib, c_fib, "k", lw=2, label="Fiber (cam4)")
    ax4.plot(r_flux_ref, c_flux_ref, "r", lw=2, label="Ref flux (cam2)")
    ax4.axvline(r90_fib, color="k", ls=":", lw=1)
    ax4.axvline(r90_flux_ref, color="r", ls=":", lw=1)
    ax4.set_xlabel("Radius [pixels]")
    ax4.set_ylabel("Cumulative photons/seconds")
    ax4.set_title(f"EE90 Ratio = {ee90_ratio:.3f}")
    ax4.legend(fontsize=8)

    plt.show()

# ----------------------------
# Write CSV
# ----------------------------
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fiber_number", "FRD_loss", "EE90_flux_loss"])
    writer.writerows(results)

print(f"Saved results to {output_csv}")
