import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

image_sizes = ["2048*2048", "4096*4096", "8192*8192"]

# X axis labels
x_titles = ["GM", "CM_CF12_Vec", "linearRowFilter +\nlinearColumnFilter\n(OpenCV)", "convolve2d\n(ArrayFire)"]

# Execution times for each image size and method
execution_times_2048 = [245, 82, 228, 368]  # GM, CM_CF12_Vec, OpenCV, ArrayFire for 2048
execution_times_4096 = [971, 306, 913, 1453]  # GM, CM_CF12_Vec, OpenCV, ArrayFire for 4096
execution_times_8192 = [3885, 1187, 3776, 5853]  # GM, CM_CF12_Vec, OpenCV, ArrayFire for 8192

execution_times = [execution_times_2048, execution_times_4096, execution_times_8192]

# Colors for each image size
colors = ['#A2CBE5', '#F4A5AE', '#A9D8B8']

# New plot layout
plt.figure(figsize=(12, 10))

bar_width = 0.25
index = range(len(x_titles))

# Create the bar plot
for i, execution_time in enumerate(execution_times):
    plt.bar([x - bar_width + (i * bar_width) for x in index], execution_time, width=bar_width, color=colors[i])

# Add labels and title
plt.title("Yürütme Süresi (µs)", fontsize=12)
plt.xlabel("Çekirdek")
plt.ylabel("Yürütme Süresi (µs)")
plt.xticks(index, x_titles)

# Create color patches for legend, starting with "Görüntü Boyutları"
patches = [mpatches.Patch(color='white', label="Görüntü Boyutları:")]  # "Görüntü Boyutları" first
patches += [mpatches.Patch(color=colors[i], label=image_sizes[i]) for i in range(len(image_sizes))]

# Add legend to the bottom, below the plot, centered, without a frame
plt.legend(handles=patches, loc='lower left', ncol=4, frameon=False, bbox_to_anchor=(-0.0, -0.2))

# Layout adjustment
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig('3rd-party.png', dpi=300, bbox_inches='tight')
plt.show()
