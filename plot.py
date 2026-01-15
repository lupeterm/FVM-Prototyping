import matplotlib.pyplot as plt
import numpy as np

threads = [1, 2, 4, 8, 16, 32]
xlabels = [str(t) for t in threads]
ldcs_mean_gc = [254, 170, 112, 150, 152, 112]
ldcm_mean_gc = [2546, 1194, 1310, 1293]
wind_mean_gc = [1898, 985, 1000, 896]

ldcs_mean_nogc = [352, 272, 180, 350, 176, 177]
ldcm_mean_nogc = [2746, 2832, 1975, 1724]
wind_mean_nogc = [3355, 1276, 1706, 1280]

ldcs_median_gc = [250, 170, 111, 145, 136, 113]
ldcm_median_gc = [2564, 1196, 1320, 1265]
wind_median_gc = [1470, 985, 1038, 863]

ldcs_median_nogc = [341, 282, 147, 350, 156, 149]
ldcm_median_nogc = [2091, 1821, 1733, 1699]
wind_median_nogc = [2304, 1292, 1433, 1214]
ldcs_cpp = 150
ldcm_cpp = 750
wind_cpp = 1100

mean_gc = [ldcs_mean_gc, ldcm_mean_gc, wind_mean_gc]
mean_nogc = [ldcs_mean_nogc, ldcm_mean_nogc, wind_mean_nogc]
median_gc = [ldcs_median_gc, ldcm_median_gc, wind_median_gc]
median_nogc = [ldcs_mean_nogc, ldcm_mean_nogc ,wind_mean_nogc]
data = [mean_gc, mean_nogc, median_gc, median_nogc]

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
names = [
    "Mean Assembly Time, GC Time Excluded",
    "Mean Assembly Time, GC Time Included",
    "Median Assembly, GC Time Excluded",
    "Median Assembly, GC Time Included",
]
yticks = [100, 150, 300, 750, 1100, 1500, 2100, 2700, 3300]

for i, ax in enumerate(axes.flat):
    d = data[i]
    # ax.axhline(y=150, color='b', linestyle='-', label="C++")
    ax.plot(np.arange(1,7), d[0], label="Julia Lid-Driven Cavity S", color='b')
    ax.scatter([1], [150], marker='$S$', color='b', label="C++ LDC-S")
    ax.plot(np.arange(1,5), d[1], label="Julia Lid-Driven Cavity M",color='g')
    ax.scatter([1], [750], marker='$M$', color='g', label="C++ LCD-M")
    ax.plot(np.arange(1,5), d[2], label="Julia WindsorBody", color= 'r')
    ax.scatter([1], [1100], marker='$W$', color='r', label="C++ WindsorBody")
   
    ax.xaxis.set_ticks([1,2,3,4,5, 6]) #set the ticks to be a
    ax.xaxis.set_ticklabels(xlabels) # change the ticks' names to x
    if i > 1:
        ax.set_xlabel("#Threads")
    ax.set_yscale("log")
    if i % 2 == 0:
        ax.set_ylabel("Matrix Assembly Time", fontsize=14)
    ax.yaxis.set_ticks(yticks) 
    ax.yaxis.set_ticklabels([f"{y}ms" for y in yticks]) # change the ticks' names to x
    ax.set_title(names[i], fontsize=18)
    ax.set_ylim([100, 3500])
    ax.yaxis.grid(True, linestyle="dashed")
    ax.set_axisbelow(True) # so that markers are drawn on top of the grid

# Create ONE global legend
handles, labels = axes[0, 0].get_legend_handles_labels()
plt.subplots_adjust(hspace=0.360, wspace=0.14)
fig.legend(handles, labels, loc="center", ncol=3, fancybox=True, shadow=True)
fig.suptitle("Cell-Based Matrix Assembly", fontsize=22)

# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()