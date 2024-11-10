import pandas as pd
import matplotlib.pyplot as plt

csv_paths = [
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/fundamental1.csv",
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/fundamental2.csv",
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/fundamental3.csv"
]
titles = ["2048*2048", "4096*4096", "8192*8192"]
colors = ["blue", "orange", "green"]

plt.figure(figsize=(8, 8))

for i in range(len(csv_paths)):
    df = pd.read_csv(csv_paths[i], encoding='utf-16', usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    
    durations = df["Metric Name"] == "Duration"
    instruction_count = df["Metric Name"] == "Executed Instructions"
    dram_throughput = df["Metric Name"] == "DRAM Throughput"
    cycles_per_ins = df["Metric Name"] == "Warp Cycles Per Executed Instruction"
    
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

    filtered_df = df[cycles_per_ins]
    
    dram_throughput_values = []

    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        dram_throughput_values.append(value)

    filtered_df = df[cycles_per_ins]
    cycles_per_ins_value = []
    
    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        cycles_per_ins_value.append(value)

    plt.subplot(4,len(csv_paths),i + 1)
    plt.bar(kernel_names,duration_values,color = "skyblue")
    plt.ylabel("Yürütme Süresi (ms)")
    plt.xlabel("Çekirdek")
    plt.ylim(0,6000)
    plt.title(titles[i])

    plt.subplot(4, len(csv_paths), i + 4)
    plt.bar(kernel_names, instruction_count_values, color="skyblue")
    plt.ylabel("Buyruk Sayısı (10\u2076)")
    plt.ylim(0,200)

    plt.subplot(4, len(csv_paths), i + 7)
    plt.bar(kernel_names, dram_throughput_values, color="skyblue")
    plt.ylabel("DRAM Verimliliği (%)")
    plt.ylim(0,100)

    plt.subplot(4, len(csv_paths), i + 10)
    plt.bar(kernel_names, cycles_per_ins_value, color="skyblue")
    plt.ylabel("Buyruk Başına Çevrim")
    plt.ylim(0,100)

plt.subplots_adjust(wspace=0.65)
plt.savefig('duration-graphic.png', dpi=300, bbox_inches='tight')
plt.show()

