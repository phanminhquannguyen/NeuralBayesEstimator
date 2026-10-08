import matplotlib.pyplot as plt
import numpy as np

# ============================================
# Data (Subcritical LBDP)
# ============================================

observations = np.array([40, 80, 160, 320, 640, 1280, 2560, 5120])

results = {
    "DeepSets SR": {
        "lambda": [0.60, 0.52, 0.46, 0.40, 0.35, 0.31, 0.27, 0.24],
        "mu":     [0.55, 0.48, 0.42, 0.37, 0.33, 0.29, 0.26, 0.23],
    },
    "CNN 1D": {
        "lambda": [0.448, 0.342, 0.273, 0.205, 0.183, 0.155, 0.105, 0.099],
        "mu":     [0.530, 0.418, 0.378, 0.264, 0.244, 0.225, 0.151, 0.126],
    }
}

# ============================================
# Font configuration (editable)
# ============================================

font_config = {
    "suptitle_size": 18,
    "title_size": 15,
    "label_size": 13,
    "tick_size": 11,
    "legend_size": 11
}

# ============================================
# Create Subplots
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fig.suptitle(
    "Subcritical LBDP: Performance of DeepSets SR and CNN 1D",
    fontsize=font_config["suptitle_size"]
)

# ---- Lambda plot ----
axes[0].plot(observations, results["DeepSets SR"]["lambda"], marker='o', label="DeepSets SR")
axes[0].plot(observations, results["CNN 1D"]["lambda"], marker='o', label="CNN 1D")

axes[0].set_title("MAE of Lambda", fontsize=font_config["title_size"])
axes[0].set_xlabel("Number of Observations", fontsize=font_config["label_size"])
axes[0].set_ylabel("MAE (Lambda)", fontsize=font_config["label_size"])
axes[0].tick_params(axis='both', labelsize=font_config["tick_size"])
axes[0].legend(fontsize=font_config["legend_size"])
axes[0].grid(True)

# ---- Mu plot ----
axes[1].plot(observations, results["DeepSets SR"]["mu"], marker='o', label="DeepSets SR")
axes[1].plot(observations, results["CNN 1D"]["mu"], marker='o', label="CNN 1D")

axes[1].set_title("MAE of Mu", fontsize=font_config["title_size"])
axes[1].set_xlabel("Number of Observations", fontsize=font_config["label_size"])
axes[1].set_ylabel("MAE (Mu)", fontsize=font_config["label_size"])
axes[1].tick_params(axis='both', labelsize=font_config["tick_size"])
axes[1].legend(fontsize=font_config["legend_size"])
axes[1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save before showing
fig.savefig("cnn_ds_sub.pdf", dpi=300)

plt.show()