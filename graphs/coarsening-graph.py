import pandas as pd
import matplotlib.pyplot as plt

csv_paths = [
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/profile1.csv",
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/profile2.csv",
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/profile3.csv"
]
titles = ["2048*2048", "4096*4096", "8192*8192"]

plt.figure(figsize=(12, 16))

for i in range(3):
    df = pd.read_csv(csv_paths[i], encoding='utf-16', usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    
    durations = df["Metric Name"] == "Duration"
    instruction_count = df["Metric Name"] == "Executed Instructions"
    
    filtered_df = df[durations]
    kernel_names = filtered_df["Kernel Name"].str.replace("_3x3", "").str.split("(").str[0]
    
    duration_values = []
    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        if row["Metric Unit"] == "msecond":
            value *= 1000
        duration_values.append(value)

    filtered_df = df[instruction_count]
    instruction_count_values = []
    for _, row in filtered_df.iterrows():
        value = int(row["Metric Value"].replace('.', ""))
        instruction_count_values.append(int(value / 1000000))

    plt.subplot(2, len(csv_paths), i + 1)
    plt.barh(kernel_names, duration_values, color="skyblue")
    plt.xlabel("Yürütme Süresi (ms)")
    plt.ylabel("Çekirdek")
    plt.title(f"{titles[i]}")
    plt.gca().invert_yaxis()

    plt.subplot(2, len(csv_paths), i + 4)
    plt.barh(kernel_names, instruction_count_values, color="skyblue")
    plt.xlabel("Buyruk Sayısı")
    plt.ylabel("Çekirdek")
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('execution-instruction-graph.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 16))

for i in range(3):
    df = pd.read_csv(csv_paths[i], encoding='utf-16', usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    
    dram_throughput = df["Metric Name"] == "DRAM Throughput"
    cycles_per_ins = df["Metric Name"] == "Warp Cycles Per Executed Instruction"

    filtered_df = df[dram_throughput]
    dram_throughput_values = []
    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        dram_throughput_values.append(value)

    filtered_df = df[cycles_per_ins]
    cycles_per_ins_value = []
    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        cycles_per_ins_value.append(value)

    plt.subplot(2, len(csv_paths), i + 1)
    plt.barh(kernel_names, dram_throughput_values, color="skyblue")
    plt.xlabel("DRAM Verimliliği (%)")
    plt.ylabel("Çekirdek")
    plt.title(f"{titles[i]}")
    plt.gca().invert_yaxis()

    plt.subplot(2, len(csv_paths), i + 4)
    plt.barh(kernel_names, cycles_per_ins_value, color="skyblue")
    plt.xlabel("Buyruk Başına Çevrim")
    plt.ylabel("Çekirdek")
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('dram-cycles-graph.png', dpi=300, bbox_inches='tight')
plt.show()
