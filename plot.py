import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===============================
# User inputs
# ===============================
file1 = os.getcwd() + "/300_ver_1/fiber_stage_pos.csv"
file2 = os.getcwd() + "/300_ver_1/fiber_frd_throughput_results.csv"

fiber_radius_mm = 0.185
x_offset_slit_stage = 30   # mm offset for fiber_id >= 256
x_offset_bundle_stage = 50   # mm offset for fiber_id >= 256

# ===============================
# Read CSV files
# ===============================
df_pos = pd.read_csv(file1)
df_loss = pd.read_csv(file2)

# -------------------------------
# Normalise Fiber IDs
# (handles 001 vs 1)
# -------------------------------
df_pos["fiber_id"]  = df_pos.iloc[:, 0].astype(str).str.lstrip("0").astype(int)
df_loss["fiber_id"] = df_loss.iloc[:, 0].astype(str).str.lstrip("0").astype(int)

# -------------------------------
# Extract required columns
# -------------------------------
df_pos = df_pos.assign(
    X = df_pos.iloc[:, 2],   # stage X
    Y = df_pos.iloc[:, 1],   # stage Y
    X_alt = df_pos.iloc[:, 3],  # long stage X
    Y_alt = 0.0                 # all fibers on Y = 0 for slit
)

# Apply X offset for fiber IDs >= 256 
df_pos.loc[df_pos["fiber_id"] >= 256, "X_alt"] += x_offset_slit_stage
df_pos.loc[df_pos["fiber_id"] >= 256, "X"] += x_offset_bundle_stage

df_loss = df_loss.assign(
    frd_loss   = df_loss.iloc[:, 1],
    flux_loss  = df_loss.iloc[:, 2] * df_loss.iloc[:, 3],
    total_loss = df_loss.iloc[:, 4]
)

# -------------------------------
# Merge on fiber ID
# -------------------------------
data = pd.merge(df_pos, df_loss, on="fiber_id", how="inner")

# ===============================
# Generic plotting function
# ===============================
def plot_fibers(
    data, value_col, title, cbar_label,
    x_col="X", y_col="Y",
    vmin=None, vmax=None, cmap=plt.cm.viridis
):

    fig, ax = plt.subplots(figsize=(8, 8))

    values = data[value_col]
    norm = plt.Normalize(
        vmin if vmin is not None else values.min(),
        vmax if vmax is not None else values.max()
    )

    for _, row in data.iterrows():
        color = cmap(norm(row[value_col]))
        circle = plt.Circle(
            (row[x_col], row[y_col]),
            radius=fiber_radius_mm,
            color=color,
            ec="black",
            lw=0.5
        )
        ax.add_patch(circle)
        ax.text(
            row[x_col], row[y_col],
            f"{row['fiber_id']}",
            ha="center", va="center",
            fontsize=6, color="white", weight="bold"
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=cbar_label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title(title)

    ax.set_xlim(data[x_col].min() - 1, data[x_col].max() + 1)
    ax.set_ylim(data[y_col].min() - 1, data[y_col].max() + 1)

    plt.show()

# ===============================
# Plot 1: FRD loss (stage positions)
# ===============================
plot_fibers(
    data,
    value_col="frd_loss",
    title="Fiber positions coloured by FRD loss",
    cbar_label="FRD loss",
    x_col="X",
    y_col="Y"
)

# ===============================
# Plot 2: Flux loss (stage positions)
# ===============================
plot_fibers(
    data,
    value_col="flux_loss",
    title="Fiber positions coloured by Flux loss",
    cbar_label="Flux loss",
    x_col="X",
    y_col="Y"
)

# ===============================
# Plot 3: Total loss (stage positions)
# ===============================
plot_fibers(
    data,
    value_col="total_loss",
    title="Fiber positions coloured by Total loss",
    cbar_label="Total loss",
    x_col="X",
    y_col="Y"
)

# ===============================
# Plot 4: Total loss (1D layout, column 4 X, Y = 0)
# ===============================
plot_fibers(
    data,
    value_col="total_loss",
    title="Fiber layout using column 4 X positions (Y = 0)",
    cbar_label="Total loss",
    x_col="X_alt",
    y_col="Y_alt"
)
