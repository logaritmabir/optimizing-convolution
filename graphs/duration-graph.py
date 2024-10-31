import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_paths = [
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/profile1.csv",
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/profile2.csv",
    "C:/Users/ben/Desktop/optimizing-convolution/graphs/profile3.csv"
]
titles = ["2048*2048", "4096*4096", "8192*8192"]
colors = ["green", "skyblue", "red"]

plt.figure(figsize=(12, 14))

all_kernel_names = []
all_metric_values = [[] for _ in range(len(titles))]

for i in range(len(csv_paths)):
    df = pd.read_csv(csv_paths[i], encoding='utf-16', usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    
    durations = df["Metric Name"] == "Duration"
    filtered_df = df[durations]
    
    kernel_names = filtered_df["Kernel Name"].str.replace("_3x3", "").str.split("(").str[0]
    metric_values = []

    for _, row in filtered_df.iterrows():
        value = float(row["Metric Value"].replace(',', '.'))
        if row["Metric Unit"] == "msecond":
            value *= 1000
        metric_values.append(value)

    for j, name in enumerate(kernel_names):
        if name not in all_kernel_names:
            all_kernel_names.append(name)
        all_metric_values[i].append(metric_values[j])

bar_width = 0.3 
index = np.arange(len(all_kernel_names)) 

for i in range(len(titles)):
    plt.barh(index + (i * bar_width), all_metric_values[i], 
             color=colors[i], edgecolor='black', height=bar_width, label=titles[i])

# Grafik ayarları
plt.xlabel("Yürütme Süresi")
plt.ylabel("Çekirdek")
plt.title("Yürütme Sürelerinin Karşılaştırılması")
plt.yticks(index + (bar_width / 2), all_kernel_names)  
plt.gca().invert_yaxis() 
plt.legend(title='Görüntü Boyutu')
plt.tight_layout()
plt.savefig('duration-graphic.png', dpi=300, bbox_inches='tight')
