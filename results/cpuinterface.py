import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import pandas as pd
import seaborn as sb
import seaborn.objects as so
import sys
from math import log
from matplotlib.offsetbox import AnchoredText
file = sys.argv[1]
prefix = file.rsplit("/", 1)[1].replace(".csv", "")
df = pd.read_csv(file, skip_blank_lines=True)
df["time_ms"] = df["time_ns"] / 1000000
df["ms_normed"] = df["time_ms"] / df["cells"]
df["ns_normed"] = df["time_ns"] / df["cells"]
df["mean_normed"] = df.groupby(["cells", "julia_or_neon", "strategy", "threads"])[
        "ms_normed"
].transform("median")
df["ms_mean"] = df.groupby(["cells", "julia_or_neon", "strategy", "threads"])[
        "time_ms"
].transform("min")
df["ns_mean_normed"] = df.groupby(["cells", "julia_or_neon", "strategy", "threads"])[
        "ns_normed"
].transform("median")
prefix = "gpustrats"
cells = sorted(df[~df["cells"].isna()]["cells"].unique())
df["merged"] = df["strategy"] + df["julia_or_neon"]
df["merged"] = df["merged"].apply(lambda s: s.replace("JuNe", " (JuNe)").replace("NeoN", " (NeoN)"))
df["bandwidth"] = df["nnz"] * 24  / (df["ms_mean"] * 1000000000) 
print(df[(df["node"] == "gpu-nvidia-h200") & (df["julia_or_neon"] == "NeoN + Julia")]["cells"].unique())

df["logthreads"] = df["threads"].apply(lambda x: log(x))

sb.set_theme(rc={'figure.figsize':(20, 12)})
sb.set_style("whitegrid")
sb.color_palette("deep", 8)
nodes = df["node"].unique()

with sb.plotting_context("paper", font_scale=1.7):
    p = sb.relplot(
        # data=df[(df["threads"]==128) ],
        data=df,
        x="cells",
        y="bandwidth",
        kind="line",
        row="threads",
        col="node",
        col_order=["gpu-nvidia-h100", "gpu-nvidia-h200"],
        hue="merged",
        style="merged",
        markersize=12,
        markers=True
    )
    p.set(
        # yscale="log",
        xscale="log",
        xlabel="#Cells",
        ylabel="Matrix Assembly Bandwidth (GB/s)",
    )
    
    desired_order = [
        "Cell-Based (NeoN)",
        "Cell-Based (JuNe)",
        "Global Face-Based (NeoN)",
        "Global Face-Based (JuNe)",
        "Face-Based (NeoN)",
        "Face-Based (JuNe)",
        "Fused Face-Based (NeoN)",
        "Fused Cell-Based (NeoN)"
    ]
    handles, labels = p.axes.flat[0].get_legend_handles_labels()
    if p._legend is not None:
        p._legend.remove()
        p._legend = None
    print(f"labels: {labels}")
    by_label = dict(zip(labels, handles))

    ordered_labels = [label for label in desired_order if label in by_label]
    ordered_handles = [by_label[label] for label in ordered_labels]
    
    at = AnchoredText(
        "INTEL XEON",
        loc="upper left",
        # bbox_to_anchor=(0.975, 0.92),
        bbox_transform=p.axes.flat[0].transAxes,
        frameon=True,
        prop={"size": plt.rcParams["legend.fontsize"]},
    )
    p.axes.flat[0].add_artist(at)  
    at = AnchoredText(
        "AMD EPYC",
        loc="upper left",
        # bbox_to_anchor=(0.975, 0.92),
        bbox_transform=p.axes.flat[1].transAxes,
        frameon=True,
        prop={"size": plt.rcParams["legend.fontsize"]},
    )
    p.axes.flat[1].add_artist(at)    
    p.set_titles("")

    p.fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        # title="Fused Strategy",
        frameon=False,
    )
    p.fig.subplots_adjust(
        left=0.07,
        right=0.98,
        # bottom=0.18,
        top=0.78,
        # wspace=0.1,
    )

    plt.savefig(f"cpu_interface_strategies.svg")
from pprint import pprint
pprint(sorted(df[(df["threads"]==1) & (df["strategy"] == "Fused Cell-Based")]["ms_mean"].unique()))
pprint(sorted(df[(df["threads"]==2) & (df["strategy"] == "Fused Cell-Based")]["ms_mean"].unique()))
