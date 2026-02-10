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
df_thr = pd.read_csv(file2)
df_corr = pd.read_csv(file3)

# ===============================
# Normalise Fiber IDs
# ===============================
df_pos["fiber_id"]  = df_pos.iloc[:, 0].astype(str).str.lstrip("0").astype(int)
df_thr["fiber_id"] = df_thr.iloc[:, 0].astype(str).str.lstrip("0").astype(int)
df_corr["fiber_id"] = df_corr.iloc[:, 0].astype(str).str.lstrip("0").astype(int)

# ===============================
# Extract required columns
# ===============================
df_pos["X"]     = df_pos.iloc[:, 2]
df_pos["Y"]     = df_pos.iloc[:, 1]
df_pos["X_alt"] = df_pos.iloc[:, 3]
df_pos["Y_alt"] = 0.0 # slit side all on the same y-location

df_corr["offset"] = df_corr.iloc[:, 1] / 1000  # converted to mm
df_pos["X_alt"] += -df_corr["offset"] # apply the corrections based on the cam3 postion extraction

# Stage offsets for RHS bundle, because stages need to be moved for RHS fibers i.e. index >=256
df_pos.loc[df_pos["fiber_id"] >= 256, "X_alt"] += x_offset_slit_stage
# df_pos.loc[df_pos["fiber_id"] >= 256, "Y_alt"] += 0.3 # move them in y to identify in plot
df_pos.loc[df_pos["fiber_id"] >= 256, "X"]     += x_offset_bundle_stage

df_thr["frd_throughput"]   = df_thr.iloc[:, 1]
df_thr["flux_throughput"]  = df_thr.iloc[:, 4]
df_thr["total_throughput"] = df_thr["frd_throughput"]*df_thr["flux_throughput"]

# ===============================
# Merge
# ===============================
data = pd.merge(df_pos, df_thr, on="fiber_id", how="inner")
data["X_alt"] += -58.4 # so that centre of slit lies at 0
data["X"] += -18.89 # so that centre of object bundle lies at 0
data["Y"] += -12.05 # so that centre of object bundle lies at 0

# ==========================================================
# COMMON SETTINGS FOR BROKEN-X PLOTS (1–3)
# ==========================================================
xlims   = [(-19.5, -15), (-4, 4), (15, 19.5)]
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
    fig.suptitle(title, y=0.95)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label=cbar_label)
    plt.savefig("Plot_"+title+".pdf", bbox_inches='tight', dpi=150)
    
# ==========================================================
# PLOT 1 — FRD loss (BROKEN X)
# ==========================================================
broken_stage_plot(
    data["frd_throughput"],
    plt.cm.viridis,
    "FRD throughput",
    "FRD throughput"
)

# ==========================================================
# PLOT 2 — Flux loss (BROKEN X)
# ==========================================================
broken_stage_plot(
    data["flux_throughput"],
    plt.cm.viridis,
    "Flux throughput",
    "Flux throughput"
)

# ==========================================================
# PLOT 3 — Total loss (BROKEN X)
# ==========================================================
broken_stage_plot(
    data["total_throughput"],
    plt.cm.plasma,
    "Total throughput",
    "Total throughput"
)

# ==========================================================
# PLOT 4 — 1D slit layout (NO BREAK)
# ==========================================================
fig, ax = plt.subplots(figsize=(60, 5))
norm = plt.Normalize(data["total_throughput"].min(), data["total_throughput"].max())

for _, row in data.iterrows():
    ax.add_patch(plt.Circle(
        (row["X_alt"], row["Y_alt"]),
        fiber_radius_mm,
        facecolor=plt.cm.viridis(norm(row["total_throughput"])),
        edgecolor="black",
        lw=0.5
    ))
    ax.text(
        row["X_alt"], row["Y_alt"],
        f"{int(row['fiber_id'])}",
        ha="center", va="center",
        fontsize=5, color="white", weight="bold"
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Total throughput")

ax.set_title("Fiber layout on slit")
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Y")
# ax.set_aspect("equal")
ax.set_xlim(data["X_alt"].min() - 1, data["X_alt"].max() + 1)
ax.set_xlim(data["X_alt"].min() - 1, data["X_alt"].max() + 1)
ax.set_ylim(-0.2, 0.2)
plt.tight_layout()
plt.savefig("Plot_Slit.pdf", bbox_inches='tight', dpi=150)

# plt.show()
out_cols = [
    "fiber_id",
    "X",
    "Y",
    "X_alt",
    "frd_throughput",
    "flux_throughput",
]

out = data[out_cols].copy()

out = out.rename(columns={
    "fiber_id": "ID",
    "X": "sky_x",
    "Y": "sky_y",
    "X_alt": "slit_x",
})

out_file = os.path.join(os.getcwd(), "final_data.csv")
out.to_csv(out_file, index=False, float_format="%.2f")

print(f"Saved {out_file}")
