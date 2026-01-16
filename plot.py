import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
threads = [1, 2, 4, 8, 16, 32]
xlabels = [str(t) for t in threads]
# mean exclusive GC time
ldcm_median_gc = [791, 677, 515, 502, 442, 444]
ldcs_median_gc = [78,   54,  49,  46,  41,  41]
wind_median_gc = [418, 418, 342, 336, 348, 340]

# mean inclusive GC time
ldcm_mean_nogc = [1740, 1687, 1102, 1001, 932, 865]
ldcs_mean_nogc = [278,   286,  111,  140,  91,  100]
wind_mean_nogc = [1415, 1194,  906,   661, 627, 615]

# median inclusive GC time
ldcm_median_nogc = [943, 879, 690, 638, 946, 800]
ldcs_median_nogc = [ 90,  67,  55,  64,  51,  51]
wind_median_nogc = [773, 565, 525, 437, 462, 431]


ldcs_cpp = 150
ldcm_cpp = 1100
wind_cpp = 750



median_gc = [ldcs_median_gc, ldcm_median_gc, wind_median_gc]
mean_nogc = [ldcs_mean_nogc, ldcm_mean_nogc, wind_mean_nogc]
median_nogc = [ldcs_median_nogc, ldcm_median_nogc ,wind_median_nogc]
data = [median_gc, mean_nogc, median_nogc]

fig, axes = plt.subplots(3,1, figsize=(6, 6))
names = [
    "Median Assembly Time, GC Time Excluded",
    "Mean Assembly Time, GC Time Included",
    "Median Assembly, GC Time Included",
]
yticks = [100, 150, 300, 750, 1100, 1500, 2100]

for i, ax in enumerate(axes.flat):
    d = data[i]
    ax.plot(np.arange(1,7), d[0], label="Julia Lid-Driven Cavity S", color='b')
    ax.scatter([1], ldcs_cpp, marker='$S$', color='b', label="C++ LDC-S")
    ax.plot(np.arange(1,7), d[1], label="Julia Lid-Driven Cavity M",color='g')
    ax.scatter([1], ldcm_cpp, marker='$M$', color='g', label="C++ LCD-M")
    ax.plot(np.arange(1,7), d[2], label="Julia WindsorBody", color= 'r')
    ax.scatter([1], wind_cpp, marker='$W$', color='r', label="C++ WindsorBody")
   
    ax.xaxis.set_ticks([1,2,3,4,5, 6]) #set the ticks to be a
    ax.xaxis.set_ticklabels(xlabels) # change the ticks' names to x
    if i > 1:
        ax.set_xlabel("#Threads")
    ax.set_ylabel("Matrix Assembly Time", fontsize=14)
    # ax.yaxis.set_ticks(yticks) 
    # ax.yaxis.set_ticklabels([f"{y}ms" for y in yticks]) # change the ticks' names to x
    ax.set_title(names[i], fontsize=18)
    # ax.set_ylim([40, 3500])
    ax.yaxis.grid(True, linestyle="dashed")
    ax.set_axisbelow(True) # so that markers are drawn on top of the grid

# Create ONE global legend
handles, labels = axes[0].get_legend_handles_labels()
plt.subplots_adjust(hspace=0.2)
fig.legend(handles, labels, loc="lower center",bbox_to_anchor=(0.5, 0.04), ncol=6, fancybox=True, shadow=True)
fig.suptitle("Cell-Based Matrix Assembly", fontsize=22)

# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

