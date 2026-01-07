import os
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

# ----------------------------
# Helper functions
# ----------------------------
def corner_medians(image, size=200):
    ny, nx = image.shape
    corners = [
        image[0:size, 0:size],
        image[0:size, nx-size:nx],
        image[ny-size:ny, 0:size],
        image[ny-size:ny, nx-size:nx]
    ]
    return [np.nanmedian(c) for c in corners]


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
    """Simple quadratic cumulative model."""
    return A * r**2

# ----------------------------
# Reference image analysis
# ----------------------------
with fits.open(reference_fits) as hdul:
    ref_data = hdul[0].data.astype(float)

ref_bg = np.median(corner_medians(ref_data, corner_size))
ref_sub = ref_data - ref_bg

x_ref, y_ref = find_centroid_2dg(ref_sub)
r_ref, c_ref = cumulative_radial_profile(ref_sub, x_ref, y_ref)

# Normalise
c_ref_norm = c_ref / np.nanmax(c_ref)

# 75% intensity radius
r75 = np.interp(0.75, c_ref_norm, r_ref)

# Fit ONLY below 75%
fit_mask = c_ref_norm <= 0.75

popt, _ = curve_fit(
    quad_model,
    r_ref[fit_mask],
    c_ref_norm[fit_mask],
    p0=(1.0 / r75**2)
)

A_fit = popt[0]

# Radius where quadratic reaches 100%
r_cross = np.sqrt(1.0 / A_fit)

# ----------------------------
# Plot reference image
# ----------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 5), constrained_layout=True
)

# Reference image
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

# Reference cumulative + fit
ax2.plot(r_ref, c_ref_norm, "k", lw=2, label="Reference")
r_fit = np.linspace(0, r_ref.max(), 500)
ax2.plot(r_fit, quad_model(r_fit, A_fit), "r--", lw=2, label=r"Fit: $A r^2$")
ax2.axvline(r_cross, color="blue", ls=":", lw=2,
            label=f"100% at {r_cross:.1f} px")

ax2.set_xlabel("Radius [pixels]")
ax2.set_ylabel("Normalised cumulative counts")
ax2.set_ylim(0, 1.05)
ax2.set_title("Reference Cumulative Profile")
ax2.legend(fontsize=8)

plt.show()

# ----------------------------
# Main loop over individual fiber images
# ----------------------------
fits_files = sorted(f for f in os.listdir(fits_dir) if f.lower().endswith(".fits"))

for fname in fits_files:
    path = os.path.join(fits_dir, fname)

    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)

    bg = np.median(corner_medians(data, corner_size))
    data_sub = data - bg

    x_cen, y_cen = find_centroid_2dg(data_sub)
    radius, cum_counts = cumulative_radial_profile(data_sub, x_cen, y_cen)
    cum_norm = cum_counts / np.nanmax(cum_counts)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )

    # Image
    im = ax1.imshow(
        data_sub,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(data_sub, 5),
        vmax=np.percentile(data_sub, 99)
    )
    ax1.plot(x_cen, y_cen, "+", color="red", markersize=18, mew=2)
    ax1.set_title(f"{fname}\n2D Gaussian Centroid")
    ax1.set_xlabel("X [pixels]")
    ax1.set_ylabel("Y [pixels]")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Cumulative profiles
    ax2.plot(radius, cum_norm, color="gray", alpha=0.6, label="Fiber: "+fname.split("_")[0])
    ax2.plot(r_ref, c_ref_norm, "k", lw=2, label="Reference")
    ax2.plot(r_fit, quad_model(r_fit, A_fit), "r--", lw=2, label=r"Fit: $A r^2$")
    ax2.axvline(r_cross, color="blue", ls=":", lw=2,
                label=f"100% at {r_cross:.1f} px")

    ax2.set_xlabel("Radius [pixels]")
    ax2.set_ylabel("Normalised cumulative counts")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Cumulative Radial Profile")
    ax2.legend(fontsize=8)

    plt.show()
