import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_execution_times(image_sizes, x_titles, execution_times, colors, output_file):
    plt.figure(figsize=(12, 9))
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
    titles = ["1536*1536", "3072*3072", "4608*4608"]
    x_titles = ["GM", "CM_CF12_Vec", "linearRowFilter +\nlinearColumnFilter\n(OpenCV)", "convolve2d\n(ArrayFire)"]
    execution_times = [
        [143, 48.64, 132.32, 206],  # 1536
        [565.54, 175.23, 512.19, 821.55],  # 3072
        [1230, 396.22, 1173.95, 1840]  # 4608
    ]
    colors = ['#A2CBE5', '#F4A5AE', '#A9D8B8']
    output_file = 'libraryFuncsComparement.png'

    plot_execution_times(titles, x_titles, execution_times, colors, output_file)

if __name__ == "__main__":
    main()
