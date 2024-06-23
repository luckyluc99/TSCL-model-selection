import numpy as np
import matplotlib.pyplot as plt
from aeon.distances import dtw_cost_matrix, dtw_alignment_path

# Create time array
n = 10
t = np.linspace(0, np.pi, n)

np.random.seed(123)

noise1 = 0.2 * np.random.randn(n)
noise2 = 0.2 * np.random.randn(n)
shift = np.pi / 4

# Create sine wave with second shifted and noise added
sine1 = np.sin(t) + noise1
sine2 = np.sin(t - shift) + noise2


plt.figure(figsize=(8, 8))

# Plot the sine waves
plt.plot(sine1, label="Sine wave 1")
plt.plot(sine2, label="Sine wave 2")

path, cost = dtw_alignment_path(sine1, sine2)
x_indices, y_indices = zip(*path)

for xi, yi in path:
    plt.plot([xi, yi], [sine1[xi], sine2[yi]], color="gray", linestyle="--", alpha=0.5)

cost_matrix = dtw_cost_matrix(sine1, sine2, window=0.2)
plt.figure(figsize=(8, 8))
plt.imshow(cost_matrix, origin="lower", cmap="Blues", interpolation="nearest")

# Overlay the alignment path on the cost matrix
path_x, path_y = zip(*path)
plt.plot(
    path_y, path_x, marker="s", color="black", markersize=12, linestyle="", linewidth=2
)
plt.axis("off")

for i in range(cost_matrix.shape[0]):
    for j in range(cost_matrix.shape[1]):
        color = "white" if (i, j) in path else "black"
        plt.text(
            j,
            i,
            f"{cost_matrix[i, j]:.2f}",
            fontsize=12,
            color=color,
            ha="center",
            va="center",
            backgroundcolor="black" if (i, j) in path else "none",
        )

plt.show()
