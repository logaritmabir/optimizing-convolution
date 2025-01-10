import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_execution_times(image_sizes, x_titles, execution_times, colors, output_file):
    plt.figure(figsize=(12, 10))
    bar_width = 0.25
    index = range(len(x_titles))

    for i, execution_time in enumerate(execution_times):
        plt.bar([x - bar_width + (i * bar_width) for x in index], execution_time, width=bar_width, color=colors[i])

    plt.title("Yürütme Süresi (µs)", fontsize=12)
    plt.xlabel("Çekirdek")
    plt.ylabel("Yürütme Süresi (µs)")
    plt.xticks(index, x_titles)

    patches = [mpatches.Patch(color='white', label="Görüntü Boyutları:")]
    patches += [mpatches.Patch(color=colors[i], label=image_sizes[i]) for i in range(len(image_sizes))]

    plt.legend(handles=patches, loc='lower left', ncol=4, frameon=False, bbox_to_anchor=(-0.0, -0.2))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    image_sizes = ["2048*2048", "4096*4096", "8192*8192"]
    x_titles = ["GM", "CM_CF12_Vec", "linearRowFilter +\nlinearColumnFilter\n(OpenCV)", "convolve2d\n(ArrayFire)"]
    execution_times = [
        [245, 82, 228, 368],  # 2048
        [971, 306, 913, 1453],  # 4096
        [3885, 1187, 3776, 5853]  # 8192
    ]
    colors = ['#A2CBE5', '#F4A5AE', '#A9D8B8']
    output_file = 'libraryFuncsComparement.png'

    plot_execution_times(image_sizes, x_titles, execution_times, colors, output_file)

if __name__ == "__main__":
    main()
