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
x_offset_slit_stage = 30     # mm offset for fiber_id >= 256
x_offset_bundle_stage = 50   # mm offset for fiber_id >= 256

# ===============================
# Read CSV files
# ===============================
df_pos = pd.read_csv(file1)
df_loss = pd.read_csv(file2)

# -------------------------------
# Normalise Fiber IDs
# -------------------------------
df_pos["fiber_id"]  = df_pos.iloc[:, 0].astype(str).str.lstrip("0").astype(int)
df_loss["fiber_id"] = df_loss.iloc[:, 0].astype(str).str.lstrip("0").astype(int)

# -------------------------------
# Extract required columns
# -------------------------------
df_pos["X"]     = df_pos.iloc[:, 2]
df_pos["Y"]     = df_pos.iloc[:, 1]
df_pos["X_alt"] = df_pos.iloc[:, 3]
df_pos["Y_alt"] = 0.0

# Apply offsets
df_pos.loc[df_pos["fiber_id"] >= 256, "X_alt"] += x_offset_slit_stage
df_pos.loc[df_pos["fiber_id"] >= 256, "X"]     += x_offset_bundle_stage

df_loss["frd_loss"]   = df_loss.iloc[:, 1]
df_loss["flux_loss"]  = df_loss.iloc[:, 2] * df_loss.iloc[:, 3]
df_loss["total_loss"] = df_loss.iloc[:, 4]

# -------------------------------
# Merge
# -------------------------------
data = pd.merge(df_pos, df_loss, on="fiber_id", how="inner")

# ==========================================================
# PLOT 1 — FRD loss, stage positions
# ==========================================================
fig, ax = plt.subplots(figsize=(30, 3))
norm = plt.Normalize(data["frd_loss"].min(), data["frd_loss"].max())

for _, row in data.iterrows():
    ax.add_patch(plt.Circle(
        (row["X"], row["Y"]),
        fiber_radius_mm,
        color=plt.cm.viridis(norm(row["frd_loss"])),
        ec="black", lw=0.5
    ))
    ax.text(
        row["X"], row["Y"],
        f"{int(row['fiber_id'])}",
        ha="center", va="center",
        fontsize=6, color="white", weight="bold"
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
sm.set_array([])
plt.colorbar(sm, ax=ax, label="FRD loss")

ax.set_title("Fiber positions coloured by FRD loss")
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Y position (mm)")
ax.set_aspect("equal")
ax.set_xlim(data["X"].min() - 1, data["X"].max() + 1)
ax.set_ylim(data["Y"].min() - 1, data["Y"].max() + 1)

plt.show()

# ==========================================================
# PLOT 2 — Flux loss, stage positions
# ==========================================================
fig, ax = plt.subplots(figsize=(30, 3))
norm = plt.Normalize(0.0, data["flux_loss"].max())

for _, row in data.iterrows():
    ax.add_patch(plt.Circle(
        (row["X"], row["Y"]),
        fiber_radius_mm,
        color=plt.cm.viridis(norm(row["flux_loss"])),
        ec="black", lw=0.5
    ))
    ax.text(
        row["X"], row["Y"],
        f"{int(row['fiber_id'])}",
        ha="center", va="center",
        fontsize=6, color="white", weight="bold"
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Flux loss")

ax.set_title("Fiber positions coloured by Flux loss")
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Y position (mm)")
ax.set_aspect("equal")

# IMPORTANT: widened limits to include offset fibers
ax.set_xlim(data["X"].min() - 1, data["X"].max() + 1)
ax.set_ylim(data["Y"].min() - 1, data["Y"].max() + 1)

plt.show()

# ==========================================================
# PLOT 3 — Total loss, stage positions
# ==========================================================
fig, ax = plt.subplots(figsize=(30, 3))
norm = plt.Normalize(data["total_loss"].min(), data["total_loss"].max())

for _, row in data.iterrows():
    ax.add_patch(plt.Circle(
        (row["X"], row["Y"]),
        fiber_radius_mm,
        color=plt.cm.plasma(norm(row["total_loss"])),
        ec="black", lw=0.5
    ))
    ax.text(
        row["X"], row["Y"],
        f"{int(row['fiber_id'])}",
        ha="center", va="center",
        fontsize=6, color="white", weight="bold"
    )

sm = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Total loss")

ax.set_title("Fiber positions coloured by Total loss")
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Y position (mm)")
ax.set_aspect("equal")
ax.set_xlim(data["X"].min() - 1, data["X"].max() + 1)
ax.set_ylim(data["Y"].min() - 1, data["Y"].max() + 1)

plt.show()

# ==========================================================
# PLOT 4 — 1D slit layout
# ==========================================================
fig, ax = plt.subplots(figsize=(60, 3))
norm = plt.Normalize(data["total_loss"].min(), data["total_loss"].max())

for _, row in data.iterrows():
    ax.add_patch(plt.Circle(
        (row["X_alt"], row["Y_alt"]),
        fiber_radius_mm,
        color=plt.cm.viridis(norm(row["total_loss"])),
        ec="black", lw=0.5
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
ax.set_ylim(-0.2, 0.2)

plt.show()
