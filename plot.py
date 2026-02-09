import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===============================
# User inputs
# ===============================
file1 = os.getcwd() + "/300_ver_1/fiber_stage_pos.csv"
file2 = os.getcwd() + "/300_ver_1/fiber_frd_throughput_results.csv"
file3 = os.getcwd() + "/300_ver_1/long_stage_corrections.csv"

fiber_radius_mm = 0.185
x_offset_slit_stage   = 8.84
x_offset_bundle_stage = 35

# ===============================
# Read CSV files
# ===============================
df_pos  = pd.read_csv(file1)
df_loss = pd.read_csv(file2)
df_corr = pd.read_csv(file3)

# ===============================
# Normalise Fiber IDs
# ===============================
df_pos["fiber_id"]  = df_pos.iloc[:, 0].astype(str).str.lstrip("0").astype(int)
df_loss["fiber_id"] = df_loss.iloc[:, 0].astype(str).str.lstrip("0").astype(int)
df_corr["fiber_id"] = df_loss.iloc[:, 0].astype(str).str.lstrip("0").astype(int)

# ===============================
# Extract required columns
# ===============================
df_pos["X"]     = df_pos.iloc[:, 2]
df_pos["Y"]     = df_pos.iloc[:, 1]
df_pos["X_alt"] = df_pos.iloc[:, 3]
df_pos["Y_alt"] = 0.0

df_corr["offset"] = df_corr.iloc[:, 1] / 1000  # mm
df_pos["X_alt"] += -df_corr["offset"]

# Stage offsets for RHS bundle
df_pos.loc[df_pos["fiber_id"] >= 256, "X_alt"] += x_offset_slit_stage
df_pos.loc[df_pos["fiber_id"] >= 256, "Y_alt"] += 1
df_pos.loc[df_pos["fiber_id"] >= 256, "X"]     += x_offset_bundle_stage

df_loss["frd_loss"]   = df_loss.iloc[:, 1]
df_loss["flux_loss"]  = 1 - ((1 - df_loss.iloc[:, 2]) / df_loss.iloc[:, 3])
df_loss["total_loss"] = 1 - (1 - df_loss["frd_loss"]) * (1 - df_loss["flux_loss"])

# ===============================
# Merge
# ===============================
data = pd.merge(df_pos, df_loss, on="fiber_id", how="inner")

# ==========================================================
# COMMON SETTINGS FOR BROKEN-X PLOTS (1–3)
# ==========================================================
xlims   = [(0, 4), (15, 23), (34, 38)]
widths = [xmax - xmin for xmin, xmax in xlims]

ymin  = data["Y"].min() - 1
ymax  = data["Y"].max() + 1
yspan = ymax - ymin

def broken_stage_plot(values, cmap, title, cbar_label):
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        sharey=True,
        figsize=(12, 4),
        gridspec_kw={"width_ratios": widths, "wspace": 0.05},
        constrained_layout=True
    )
    axes = [ax1, ax2, ax3]

    norm = plt.Normalize(values.min(), values.max())

    for ax, (xmin, xmax) in zip(axes, xlims):
        sub = data[(data["X"] >= xmin) & (data["X"] <= xmax)]

        for _, row in sub.iterrows():
            ax.add_patch(plt.Circle(
                (row["X"], row["Y"]),
                fiber_radius_mm,
                facecolor=cmap(norm(row[values.name])),
                edgecolor="black",
                lw=0.5
            ))
            ax.text(
                row["X"], row["Y"],
                f"{int(row['fiber_id'])}",
                ha="center", va="center",
                fontsize=6, color="white", weight="bold"
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_box_aspect(yspan / (xmax - xmin))

    # Spines and ticks
    ax1.spines.right.set_visible(False)
    ax2.spines.left.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax3.spines.left.set_visible(False)
    ax2.tick_params(labelleft=False)
    ax3.tick_params(labelleft=False)

    # Break markers
    d = 0.015
    for left, right in [(ax1, ax2), (ax2, ax3)]:
        left.plot((1-d, 1+d), (-d, +d), transform=left.transAxes, color="k", clip_on=False)
        left.plot((1-d, 1+d), (1-d, 1+d), transform=left.transAxes, color="k", clip_on=False)
        right.plot((-d, +d), (-d, +d), transform=right.transAxes, color="k", clip_on=False)
        right.plot((-d, +d), (1-d, 1+d), transform=right.transAxes, color="k", clip_on=False)

    ax1.set_ylabel("Y position (mm)")
    fig.supxlabel("X position (mm)")
    fig.suptitle(title, y=1.02)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label=cbar_label)

    plt.show()

# ==========================================================
# PLOT 1 — FRD loss (BROKEN X)
# ==========================================================
broken_stage_plot(
    data["frd_loss"],
    plt.cm.viridis,
    "Fiber positions coloured by FRD loss",
    "FRD loss"
)

# ==========================================================
# PLOT 2 — Flux loss (BROKEN X)
# ==========================================================
broken_stage_plot(
    data["flux_loss"],
    plt.cm.viridis,
    "Fiber positions coloured by Flux loss",
    "Flux loss"
)

# ==========================================================
# PLOT 3 — Total loss (BROKEN X)
# ==========================================================
broken_stage_plot(
    data["total_loss"],
    plt.cm.plasma,
    "Fiber positions coloured by Total loss",
    "Total loss"
)

# ==========================================================
# PLOT 4 — 1D slit layout (NO BREAK)
# ==========================================================
fig, ax = plt.subplots(figsize=(60, 10))
norm = plt.Normalize(data["total_loss"].min(), data["total_loss"].max())

for _, row in data.iterrows():
    ax.add_patch(plt.Circle(
        (row["X_alt"], row["Y_alt"]),
        fiber_radius_mm,
        facecolor=plt.cm.viridis(norm(row["total_loss"])),
        edgecolor="black",
        lw=0.5
    ))
    ax.text(
        row["X_alt"], row["Y_alt"],
        f"{int(row['fiber_id'])}",
        ha="center", va="center",
        fontsize=6, color="white", weight="bold"
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Total loss")

ax.set_title("Fiber layout using column 4 X positions (Y = 0)")
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Y")
ax.set_aspect("equal")
ax.set_xlim(data["X_alt"].min() - 1, data["X_alt"].max() + 1)
ax.set_ylim(-2, 2)

plt.show()
