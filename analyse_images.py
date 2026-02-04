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
output_csv = "fiber_frd_throughput_results.csv"

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


def px_to_f(px, r_cross, f_ref):
    px = np.asarray(px)
    return (r_cross / px) * f_ref


def f_to_px(f, r_cross, f_ref):
    f = np.asarray(f)
    return (r_cross / f) * f_ref
    

def add_fratio_axis(ax, r_cross, f_ref):

    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda px: px_to_f(px, r_cross, f_ref),
            lambda f:  f_to_px(f, r_cross, f_ref),
        )
    )
    secax.set_xlabel("f-ratio")
    return secax
    

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

f_2_273_px = f_to_px(2.273, r_cross, fratio_ref)
cum_counts_2_273_ref = np.interp(f_2_273_px, r_ref, c_ref)
# ----------------------------
# Reference plot
# ----------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

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
secax = add_fratio_axis(ax2, r_cross, fratio_ref)
secax.set_xticks([1, 2.3, 3, 4.2, 7, 20])
secax.set_xticklabels([f"f/{i}" for i in [1, 2.3, 3, 4.2, 7, 20]])

ax3.plot(r_ref, c_ref, "k", lw=2, label="Reference")
ax3.axvline(f_2_273_px, color="blue", ls=":", lw=2,label=f"Cumulative counts @ f/2.273 = {cum_counts_2_273_ref:.3e}")
ax3.set_xlabel("Radius [pixels]")
ax3.set_ylabel("Cumulative counts")
ax3.set_ylim(0,)
# ax2.set_title("Reference Cumulative Profile")
ax3.legend(fontsize=8)
secax = add_fratio_axis(ax3, r_cross, fratio_ref)
secax.set_xticks([1, 2.3, 3, 4.2, 7, 20])
secax.set_xticklabels([f"f/{i}" for i in [1, 2.3, 3, 4.2, 7, 20]])


# plt.show()
plt.savefig("reference.pdf", bbox_inches='tight', dpi=150)


# ----------------------------
# Main loop
# ----------------------------
results = []

fits_files = sorted(f for f in os.listdir(fits_dir) if f.lower().endswith(".fits"))

C90_flux_ref_origin = None

for fname in fits_files:
    fiber_path = os.path.join(fits_dir, fname)
    ref_flux_path = os.path.join(
        fits_dir.replace("cam4", "cam2"),
        fname.replace("cam4", "cam2")
    )

    with fits.open(fiber_path) as hdul:
        fiber_data = hdul[0].data.astype(float)

    with fits.open(ref_flux_path) as hdul:
        ref_flux_data = hdul[0].data.astype(float)

    fiber_sub = fiber_data - np.mean(corner_means(fiber_data, corner_size))
    ref_flux_sub = ref_flux_data - np.mean(corner_means(ref_flux_data, corner_size))

    x_fib, y_fib = find_centroid_2dg(fiber_sub)
    x_flux_ref, y_flux_ref = find_centroid_2dg(ref_flux_sub)

    r_fib, c_fib = cumulative_radial_profile(fiber_sub, x_fib, y_fib)
    r_flux_ref, c_flux_ref = cumulative_radial_profile(ref_flux_sub, x_flux_ref, y_flux_ref)

    c_fib_norm = c_fib / np.nanmax(c_fib)
    c_flux_ref_norm = c_flux_ref / np.nanmax(c_flux_ref)

    r90_flux_ref = np.interp(0.90, c_flux_ref_norm, r_flux_ref)

    C90_flux_ref = np.interp(r90_flux_ref, r_flux_ref, c_flux_ref)
    if C90_flux_ref_origin is None: # first iteration
        C90_flux_ref_origin = C90_flux_ref
    
    reference_flux_ratio = C90_flux_ref/C90_flux_ref_origin
    
    cum_counts_2_273_fib = np.interp(f_2_273_px, r_fib, c_fib)
    f_2_273_ratio = cum_counts_2_273_fib/cum_counts_2_273_ref
    flux_loss = 1.0 - f_2_273_ratio

    frac_at_rcross = np.interp(r_cross, r_fib, c_fib_norm)
    frd = 1.0 - frac_at_rcross

    fiber_number = fname.split("_")[0]
    results.append((fiber_number, frd, flux_loss, reference_flux_ratio))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

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
    ax2.plot(r_ref, c_ref_norm, "k", lw=2, label="Direct Reference")
    ax2.axvline(r_cross, color="blue", ls=":", lw=2)
    ax2.axhline(frac_at_rcross, color="green", ls=":", lw=2,
                label=f"FRD = {frac_at_rcross*100:.1f}%")
    ax2.set_xlabel("Radius [pixels]")
    ax2.set_ylabel("Normalised cumulative counts")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("FRD Diagnostic")
    ax2.legend(fontsize=8)
    secax = add_fratio_axis(ax2, r_cross, fratio_ref)
    secax.set_xticks([1, 2.3, 3, 4.2, 7, 20])
    secax.set_xticklabels([f"f/{i}" for i in [1, 2.3, 3, 4.2, 7, 20]])
    

    ax3.plot(r_fib, c_fib, color="gray", lw=2, label="Fiber")
    ax3.plot(r_ref, c_ref, "k", lw=2, label="Direct Reference")
    ax3.axvline(f_2_273_px, color="blue", ls=":", lw=2, label=f"Flux throughput = {f_2_273_ratio*100:.1f}%")

    ax3.set_xlabel("Radius [pixels]")
    ax3.set_ylabel("Cumulative Counts")
    ax3.set_ylim(0,)
    
    ax3.legend(fontsize=8)
    secax = add_fratio_axis(ax3, r_cross, fratio_ref)
    secax.set_xticks([1, 2.3, 3, 4.2, 7, 20])
    secax.set_xticklabels([f"f/{i}" for i in [1, 2.3, 3, 4.2, 7, 20]])

    # plt.show()
    plt.savefig(f"{fname.split(".")[0]}.pdf", bbox_inches='tight', dpi=150)
    
# ----------------------------
# Write CSV
# ----------------------------
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fiber_number", "FRD", "Flux_loss","Ref_flux"])
    writer.writerows(results)

print(f"Saved results to {output_csv}")
