import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

csv_paths = [
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/datas/imsize1536.csv",
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/datas/imsize3072.csv",
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/datas/imsize4608.csv"
]
titles = ["1536*1536", "3072*3072", "4608*4608"]

def format_kernel_names(kernel_names):
    updated_kernel_names = []
    for name in kernel_names:
        try:
            name_parts = name.split()
            function_part = name_parts[1]
            if '<' in function_part and '>' in function_part:
                func_name, param = function_part.split('<')
                param = param.split('>')[0]
                base_name, extension = func_name.split('_', 1)
                if '_Vec' in func_name:
                    extension = extension.split('_')[1]
                    updated_kernel_names.append(f"{base_name}_CF{param}_{extension}")
                else:
                    if (param == '1'):
                        updated_kernel_names.append(f"{base_name}")
                    else:
                        updated_kernel_names.append(f"{base_name}_CF{param}")
        except Exception as e:
            updated_kernel_names.append("name")
    return updated_kernel_names

def process_data(csv_path, size_title):
    df = pd.read_csv(csv_path, encoding='utf-16', usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    kernel_names = df[df["Metric Name"] == "Duration"]["Kernel Name"]
    kernel_names = format_kernel_names(kernel_names)
    durations = df[df["Metric Name"] == "Duration"]
    duration_values = []
    for _, row in durations.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        if row["Metric Unit"] == "ms":
            value *= 1000
        duration_values.append(value)
    instructions = df[df["Metric Name"] == "Executed Instructions"]
    instruction_values = []
    for _, row in instructions.iterrows():
        value = int(row["Metric Value"].replace('.', ""))
        instruction_values.append(int(value / 1000000))
    cycles = df[df["Metric Name"] == "Warp Cycles Per Executed Instruction"]
    cycle_values = []
    for _, row in cycles.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        cycle_values.append(value)
    return kernel_names, duration_values, instruction_values, cycle_values, size_title

all_data = []
for i, csv_path in enumerate(csv_paths):
    all_data.append(process_data(csv_path, titles[i]))

kernel_set = [kernel for data in all_data for kernel in data[0]]
kernel_set = sorted(set(kernel_set), key=lambda x: kernel_set.index(x))
x = np.arange(len(kernel_set))
width = 0.25

fig, ax = plt.subplots(3, 1, figsize=(14, 22))
metrics = ["Yürütme Süresi (µs)", "Buyruk Sayısı (10⁶)", "Buyruk Başına Çevrim"]
colors = ['#A2CBE5', '#F4A5AE', '#A9D8B8']

for metric_idx, ax in enumerate(ax):
    for group_idx, data in enumerate(all_data):
        kernel_names, duration_values, instruction_values, cycle_values, size_title = data
        metric_values = [duration_values, instruction_values, cycle_values][metric_idx]
        values = [metric_values[kernel_names.index(kernel)] if kernel in kernel_names else 0 for kernel in kernel_set]
        ax.bar(x + group_idx * width, values, width, color=colors[group_idx])
    ax.set_xticks(x + width)
    ax.set_xticklabels(kernel_set, rotation=45, ha='right')
    ax.set_ylabel(metrics[metric_idx])
    ax.set_xlabel("Çekirdek")
    ax.set_title(metrics[metric_idx])

patches = [mpatches.Patch(color='white', label="Görüntü Boyutları:")]
patches += [mpatches.Patch(color=colors[i], label=titles[i]) for i in range(len(titles))]

plt.legend(handles=patches, loc='lower left', ncol=4, frameon=False, bbox_to_anchor=(-0.0, -0.3))
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig('coarseningEffect.png', dpi=300, bbox_inches='tight')
plt.show()
