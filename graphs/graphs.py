import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/ben/Desktop/optimizing-convolution/graphs/profile.csv", encoding='utf-16',
                 usecols=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])

durations = df["Metric Name"] == "Duration" # ret -> bool array

filtered_df = df[durations] # ret -> selected rows

kernel_names = filtered_df["Kernel Name"].str.replace("_3x3", "").str.split("(").str[0]
metric_values = []

for _, row in filtered_df.iterrows():
    value = float(row["Metric Value"].replace(',', '.'))
    
    if row["Metric Unit"] == "msecond":
        value *= 1000
    
    metric_values.append(value)


plt.figure(figsize=(12, 8))
plt.barh(kernel_names, metric_values, color="skyblue")
plt.xlabel("Metric Value (Duration)")
plt.ylabel("Kernel Name")
plt.title("Duration Graph")
plt.gca().invert_yaxis()
plt.show()
