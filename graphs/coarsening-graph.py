import pandas as pd
import matplotlib.pyplot as plt

csv_paths = [
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/uchar/all-2048-uchar.csv",
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/uchar/all-4096-uchar.csv",
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/uchar/all-8192-uchar.csv"
]
titles = ["2048*2048", "4096*4096", "8192*8192"]

plt.figure(figsize=(16, 24))

for i in range(3):
    df = pd.read_csv(csv_paths[i], encoding='utf-16', usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    
    # Yürütme Süresi
    durations = df["Metric Name"] == "Duration"
    filtered_df = df[durations]
    kernel_names = filtered_df["Kernel Name"].str.replace("_3x3", "").str.split("(").str[0]
    
    duration_values = []
    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        if row["Metric Unit"] == "msecond":
            value *= 1000
        duration_values.append(value)

    # Buyruk Sayısı
    instruction_count = df["Metric Name"] == "Executed Instructions"
    filtered_df = df[instruction_count]
    instruction_count_values = []
    for _, row in filtered_df.iterrows():
        value = int(row["Metric Value"].replace('.', ""))
        instruction_count_values.append(int(value / 1000000))

    # Buyruk Başına Çevrim
    cycles_per_ins = df["Metric Name"] == "Warp Cycles Per Executed Instruction"
    filtered_df = df[cycles_per_ins]
    cycles_per_ins_value = []
    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        cycles_per_ins_value.append(value)

    # Yürütme Süresi Grafiği
    plt.subplot(3, 3, i + 1)
    plt.barh(kernel_names, duration_values, color="skyblue")
    plt.xlabel("Yürütme Süresi (µs)")
    plt.ylabel("Çekirdek")
    plt.title("Görüntü Boyutu : " + titles[i], fontsize = 10)
    plt.gca().invert_yaxis()

    # Buyruk Sayısı Grafiği
    plt.subplot(3, 3, i + 4)
    plt.barh(kernel_names, instruction_count_values, color="skyblue")
    plt.xlabel("Buyruk Sayısı (M)")
    plt.ylabel("Çekirdek")
    plt.gca().invert_yaxis()

    # Buyruk Başına Çevrim Grafiği
    plt.subplot(3, 3, i + 7)
    plt.barh(kernel_names, cycles_per_ins_value, color="skyblue")
    plt.xlabel("Buyruk Başına Çevrim")
    plt.ylabel("Çekirdek")
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('combined-graph.png', dpi=300, bbox_inches='tight')
plt.show()
