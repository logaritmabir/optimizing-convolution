import matplotlib.pyplot as plt

image_sizes = ["2048*2048", "4096*4096", "8192*8192"]

x_titles = ["CM_CF12_Vec","OpenCV\n(linearRowFilter +\nlinearColumnFilter)","ArrayFire\n(convolve2d)"]
x_titles_base = ["GM","OpenCV\n(linearRowFilter +\nlinearColumnFilter)","ArrayFire\n(convolve2d)"]
array_fire = [368, 1453, 5853]
opencv = [228, 913, 3776]
custom = [82, 306, 1187]

execution_times_2048 = [82, 228, 368]
execution_times_4096 = [306, 913, 1453]
execution_times_8192 = [1187, 3776, 5853]

execution_times_2048_base = [245, 228, 368]
execution_times_4096_base = [971, 913, 1453]
execution_times_8192_base = [3885, 3776, 5853]

execution_times = [execution_times_2048,execution_times_4096,execution_times_8192]
execution_times_base = [execution_times_2048_base,execution_times_4096_base,execution_times_8192_base]
plt.figure(figsize=(16, 8))

for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.title("Görüntü Boyutu : " + image_sizes[i], fontsize = 10)
    plt.bar(x_titles_base, execution_times_base[i], color = "skyblue")
    plt.xlabel("Çekirdek")
    plt.ylabel("Yürütme Süresi (µs)")
    plt.subplot(2, 3, i+4)
    plt.bar(x_titles, execution_times[i], color = "skyblue")
    plt.xlabel("Çekirdek")
    plt.ylabel("Yürütme Süresi (µs)")

plt.savefig('3rd-party-comparision.png', dpi=300, bbox_inches='tight')
plt.show()