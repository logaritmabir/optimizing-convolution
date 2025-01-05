import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

csv_paths = [
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/uchar/reports/all-2048-uchar.csv",
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/uchar/reports/all-4096-uchar.csv",
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/uchar/reports/all-8192-uchar.csv"
]
titles = ["2048*2048", "4096*4096", "8192*8192"]

# Verileri işlemek için fonksiyon tanımlama
def process_data(csv_path, size_title):
    df = pd.read_csv(csv_path, encoding='utf-16', usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    
    # Kernel adlarını işleme
    kernel_names = df[df["Metric Name"] == "Duration"]["Kernel Name"]
    kernel_names = kernel_names.str.replace("_3x3", "").str.split("(").str[0].unique()

    # Yürütme Süresi
    durations = df[df["Metric Name"] == "Duration"]
    duration_values = []
    for _, row in durations.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        if row["Metric Unit"] == "msecond":
            value *= 1000
        duration_values.append(value)

    # Buyruk Sayısı
    instructions = df[df["Metric Name"] == "Executed Instructions"]
    instruction_values = []
    for _, row in instructions.iterrows():
        value = int(row["Metric Value"].replace('.', ""))  # "." yerine "", milyonlarla ifade ediyor
        instruction_values.append(int(value / 1000000))

    # Buyruk Başına Çevrim
    cycles = df[df["Metric Name"] == "Warp Cycles Per Executed Instruction"]
    cycle_values = []
    for _, row in cycles.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        cycle_values.append(value)

    return kernel_names, duration_values, instruction_values, cycle_values, size_title

# Verileri birleştir
all_data = []
for i, csv_path in enumerate(csv_paths):
    all_data.append(process_data(csv_path, titles[i]))

# Veriyi tek bir çubuk grafik üzerinde göstermek için işleme
kernel_set = [kernel for data in all_data for kernel in data[0]]
kernel_set = sorted(set(kernel_set), key=lambda x: kernel_set.index(x))  # Orijinal sırayı koru
x = np.arange(len(kernel_set))
width = 0.25

fig, ax = plt.subplots(3, 1, figsize=(14, 22))
metrics = ["Yürütme Süresi (µs)", "Buyruk Sayısı (10⁶)", "Buyruk Başına Çevrim"]
colors = ['#A2CBE5', '#F4A5AE', '#A9D8B8']

for metric_idx, ax in enumerate(ax):
    for group_idx, data in enumerate(all_data):
        kernel_names, duration_values, instruction_values, cycle_values, size_title = data
        metric_values = [duration_values, instruction_values, cycle_values][metric_idx]

        values = [metric_values[kernel_names.tolist().index(kernel)] if kernel in kernel_names.tolist() else 0 for kernel in kernel_set]
        
        ax.bar(x + group_idx * width, values, width, color=colors[group_idx])

    ax.set_xticks(x + width)
    ax.set_xticklabels(kernel_set, rotation=45, ha='right')
    ax.set_ylabel(metrics[metric_idx])
    ax.set_title(metrics[metric_idx])

# Renk şeridi ekleme
patches = [mpatches.Patch(color='white', label="Görüntü Boyutları:")]  # "Görüntü Boyutları" first
patches += [mpatches.Patch(color=colors[i], label=titles[i]) for i in range(len(titles))]

# Add legend to the bottom, below the plot, centered, without a frame
plt.legend(handles=patches, loc='lower left', ncol=4, frameon=False, bbox_to_anchor=(-0.0, -0.3))

# Layout adjustment
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig('coarsened-kernels.png', dpi=300, bbox_inches='tight')
plt.show()
