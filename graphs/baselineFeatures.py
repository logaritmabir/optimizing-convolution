import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

csv_paths = [
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/datas/imsize1536.csv",
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/datas/imsize3072.csv",
    "C:/Users/steam/Desktop/optimizing-convolution/graphs/datas/imsize4608.csv"
]

titles = ["1536*1536", "3072*3072", "4608*4608"]
colors = ['#A2CBE5', '#F4A5AE', '#A9D8B8']

data = {
    "duration": [],
    "instruction_count": [],
    "cycles_per_instruction": []
}

kernel_names = []

kernel_selection = ["void GM_3x3<1>(unsigned char *, unsigned char *, int, int)",
                    "void CM_3x3<1>(unsigned char *, unsigned char *, int, int)",
                    "void SM_3x3<1>(unsigned char *, unsigned char *, int, int)"
                    ]

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

def main():
    global kernel_names
    for i in range(len(csv_paths)):
        df = pd.read_csv(csv_paths[i], encoding='utf-16', usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])

        df = df[df["Kernel Name"].isin(kernel_selection)]

        durations = df["Metric Name"] == "Duration"
        instruction_count = df["Metric Name"] == "Executed Instructions"
        cycles_per_ins = df["Metric Name"] == "Warp Cycles Per Executed Instruction"

        filtered_df = df[durations]
        kernel_names = filtered_df["Kernel Name"]
        updated_kernel_names = format_kernel_names(kernel_names)
        duration_values = []

        for _, row in filtered_df.iterrows():
            value = float(row["Metric Value"].replace(',', '.'))
            if row["Metric Unit"] == "ms":
                value *= 1000
            duration_values.append(value)

        data["duration"].append(duration_values)

        filtered_df = df[instruction_count]
        instruction_count_values = []
        for _, row in filtered_df.iterrows():
            value = int(row["Metric Value"].replace('.', ""))
            instruction_count_values.append(int(value / 1000000))

        data["instruction_count"].append(instruction_count_values)

        filtered_df = df[cycles_per_ins]
        cycles_per_ins_value = []
        for _, row in filtered_df.iterrows():
            value = float(row["Metric Value"].replace(',', '.'))
            cycles_per_ins_value.append(value)

        data["cycles_per_instruction"].append(cycles_per_ins_value)

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
        plt.xticks([p + bar_width for p in x], updated_kernel_names)
        plt.title(ylabel)

    patches = [mpatches.Patch(color='white', label="Görüntü Boyutları:")]
    patches += [mpatches.Patch(color=colors[i], label=titles[i]) for i in range(len(titles))]

    plt.legend(handles=patches, loc='lower left', ncol=4, frameon=False, bbox_to_anchor=(-0.0, -0.3))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig('baselineFeatures.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()