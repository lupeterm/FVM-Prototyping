import matplotlib.pyplot as plt
import numpy as np

threads = [1, 2, 4, 8, 16, 32]
xlabels = [str(t) for t in threads]
# mean exclusive GC time
ldcs_mean_gc = [69, 56, 112, 150, 152, 112]
ldcm_mean_gc = [912, 588, 1310, 1293]
wind_mean_gc = [626, 434, 1000, 896]

# mean inclusive GC time
ldcs_mean_nogc = [179, 127, 180, 350, 176, 177]
ldcm_mean_nogc = [1204, 1422, 1975, 1724]
wind_mean_nogc = [950, 1160, 1706, 1280]

# median inclusive GC time
ldcs_median_nogc = [93, 67, 147, 350, 156, 149]
ldcm_median_nogc = [822, 590, 1733, 1699]
wind_median_nogc = [874, 654, 1433, 1214]


ldcs_cpp = 150
ldcm_cpp = 1100
wind_cpp = 750



mean_gc = [ldcs_mean_gc, ldcm_mean_gc, wind_mean_gc]
mean_nogc = [ldcs_mean_nogc, ldcm_mean_nogc, wind_mean_nogc]
median_nogc = [ldcs_median_nogc, ldcm_median_nogc ,wind_median_nogc]
data = [mean_gc, mean_nogc, median_nogc]

fig, axes = plt.subplots(3,1, figsize=(6, 6))
names = [
    "Median Assembly Time, GC Time Excluded",
    "Mean Assembly Time, GC Time Included",
    "Median Assembly, GC Time Included",
]
yticks = [100, 150, 300, 750, 1100, 1500, 2100]

for i, ax in enumerate(axes.flat):
    d = data[i]
    # ax.axhline(y=150, color='b', linestyle='-', label="C++")
    ax.plot(np.arange(1,7), d[0], label="Julia Lid-Driven Cavity S", color='b')
    ax.scatter([1], ldcs_cpp, marker='$S$', color='b', label="C++ LDC-S")
    ax.plot(np.arange(1,5), d[1], label="Julia Lid-Driven Cavity M",color='g')
    ax.scatter([1], ldcm_cpp, marker='$M$', color='g', label="C++ LCD-M")
    ax.plot(np.arange(1,5), d[2], label="Julia WindsorBody", color= 'r')
    ax.scatter([1], wind_cpp, marker='$W$', color='r', label="C++ WindsorBody")
   
    ax.xaxis.set_ticks([1,2,3,4,5, 6]) #set the ticks to be a
    ax.xaxis.set_ticklabels(xlabels) # change the ticks' names to x
    if i > 1:
        ax.set_xlabel("#Threads")
    ax.set_yscale("log")
    ax.set_ylabel("Matrix Assembly Time", fontsize=14)
    ax.yaxis.set_ticks(yticks) 
    ax.yaxis.set_ticklabels([f"{y}ms" for y in yticks]) # change the ticks' names to x
    ax.set_title(names[i], fontsize=18)
    # ax.set_ylim([40, 3500])
    ax.yaxis.grid(True, linestyle="dashed")
    ax.set_axisbelow(True) # so that markers are drawn on top of the grid

# Create ONE global legend
handles, labels = axes[0].get_legend_handles_labels()
plt.subplots_adjust(hspace=0.2)
fig.legend(handles, labels, loc="lower center",bbox_to_anchor=(0.5, 0.04), ncol=6, fancybox=True, shadow=True)
fig.suptitle("Cell-Based Matrix Assembly (nThreads > 2 not updated yet)", fontsize=22)

# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

