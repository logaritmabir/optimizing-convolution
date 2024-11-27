import pandas as pd
import matplotlib.pyplot as plt

csv_paths = [
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/uchar/base-kernels-2048-uchar.csv",
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/uchar/base-kernels-4096-uchar.csv",
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/uchar/base-kernels-8192-uchar.csv"
]
titles = ["2048*2048", "4096*4096", "8192*8192"]
colors = ["blue", "orange", "green"]


plt.figure(figsize=(10, 10))

for i in range(len(csv_paths)):
    df = pd.read_csv(csv_paths[i], encoding='utf-16', usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    
    durations = df["Metric Name"] == "Duration"
    instruction_count = df["Metric Name"] == "Executed Instructions"
    dram_throughput = df["Metric Name"] == "DRAM Throughput"
    cycles_per_ins = df["Metric Name"] == "Warp Cycles Per Executed Instruction"
    
    # Duration Metric
    filtered_df = df[durations]
    kernel_names = filtered_df["Kernel Name"].str.replace("_3x3", "").str.split("(").str[0]
    duration_values = []
    
    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        if row["Metric Unit"] == "msecond":
            value *= 1000
        duration_values.append(value)
    
    # Instruction Count Metric
    filtered_df = df[instruction_count]
    instruction_count_values = []
    for _, row in filtered_df.iterrows():
        value = int(row["Metric Value"].replace('.', ""))
        instruction_count_values.append(int(value / 1000000))
    
    # DRAM Throughput Metric
    filtered_df = df[dram_throughput]
    dram_throughput_values = []
    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        dram_throughput_values.append(value)
    
    # Cycles Per Instruction Metric
    filtered_df = df[cycles_per_ins]
    cycles_per_ins_value = []
    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        cycles_per_ins_value.append(value)
    
    # Grafik Çizimi
    plt.subplot(4, len(csv_paths), i + 1)
    plt.bar(kernel_names, duration_values, color="skyblue")
    plt.ylabel("Yürütme Süresi (µs)")
    plt.xlabel("Çekirdek")
    plt.ylim(0, 6000)
    plt.title("Görüntü Boyutu : " + titles[i], fontsize = 10)

    plt.subplot(4, len(csv_paths), i + 4)
    plt.bar(kernel_names, instruction_count_values, color="skyblue")
    plt.ylabel("Buyruk Sayısı (10\u2076)")
    plt.xlabel("Çekirdek")
    plt.ylim(0, 220)

    # plt.subplot(4, len(csv_paths), i + 7)
    # plt.bar(kernel_names, dram_throughput_values, color="skyblue")
    # plt.ylabel("DRAM Verimliliği (%)")
    # plt.ylim(0, 100)

    plt.subplot(4, len(csv_paths), i + 7)
    plt.bar(kernel_names, cycles_per_ins_value, color="skyblue")
    plt.ylabel("Buyruk Başına Çevrim")
    plt.xlabel("Çekirdek")
    plt.ylim(0, 100)


plt.subplots_adjust(wspace=0.35)
plt.subplots_adjust(hspace=0.35)
plt.savefig('duration-graphic.png', dpi=300, bbox_inches='tight')
plt.show()
