import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

csv_paths = [
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/uchar/reports/base-kernels-2048-uchar.csv",
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/uchar/reports/base-kernels-4096-uchar.csv",
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/uchar/reports/base-kernels-8192-uchar.csv"
]
titles = ["2048*2048", "4096*4096", "8192*8192"]
colors = ['#A2CBE5', '#F4A5AE', '#A9D8B8']

# Initialize data storage for each metric
data = {
    "duration": [],
    "instruction_count": [],
    "cycles_per_instruction": []
}
kernel_names = []

# Process each CSV and store data
for i in range(len(csv_paths)):
    df = pd.read_csv(csv_paths[i], encoding='utf-16', usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])

    durations = df["Metric Name"] == "Duration"
    instruction_count = df["Metric Name"] == "Executed Instructions"
    cycles_per_ins = df["Metric Name"] == "Warp Cycles Per Executed Instruction"

    # Duration Metric
    filtered_df = df[durations]
    kernel_names = filtered_df["Kernel Name"].str.replace("_3x3", "").str.split("(").str[0].unique()
    duration_values = []

    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        if row["Metric Unit"] == "msecond":
            value *= 1000
        duration_values.append(value)

    data["duration"].append(duration_values)

    # Instruction Count Metric
    filtered_df = df[instruction_count]
    instruction_count_values = []
    for _, row in filtered_df.iterrows():
        value = int(row["Metric Value"].replace('.', ""))
        instruction_count_values.append(int(value / 1000000))

    data["instruction_count"].append(instruction_count_values)

    # Cycles Per Instruction Metric
    filtered_df = df[cycles_per_ins]
    cycles_per_ins_value = []
    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        cycles_per_ins_value.append(value)

    data["cycles_per_instruction"].append(cycles_per_ins_value)

# Plotting consolidated graphs
metrics = {
    "duration": "Yürütme Süresi (µs)",
    "instruction_count": "Buyruk Sayısı (10⁶)",
    "cycles_per_instruction": "Buyruk Başına Çevrim"
}

plt.figure(figsize=(10, 14))

for idx, (metric, ylabel) in enumerate(metrics.items()):
    plt.subplot(3, 1, idx + 1)
    bar_width = 0.2
    x = range(len(kernel_names))

    for i, values in enumerate(data[metric]):
        plt.bar([p + i * bar_width for p in x], values, width=bar_width, color=colors[i])

    plt.ylabel(ylabel)
    plt.xlabel("Çekirdek")
    plt.xticks([p + bar_width for p in x], kernel_names)
    plt.title(ylabel)

# Add a color legend as a horizontal bar
patches = [mpatches.Patch(color='white', label="Görüntü Boyutları:")]  # "Görüntü Boyutları" first
patches += [mpatches.Patch(color=colors[i], label=titles[i]) for i in range(len(titles))]

# Add legend to the bottom, below the plot, centered, without a frame
plt.legend(handles=patches, loc='lower left', ncol=4, frameon=False, bbox_to_anchor=(-0.0, -0.3))

# Layout adjustment
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig('base-kernels-graphics.png', dpi=300, bbox_inches='tight')
plt.show()
