import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import pandas as pd
import seaborn as sb
import seaborn.objects as so

df = pd.read_csv("variations_cpu.csv", skip_blank_lines=True)
df["cells"] = df["case_long"].apply(lambda x: int(x.split("-")[1]))
df["ms_normed"] = df["time_mean_ms"] / df["cells"]

M= {
    "PrecalculatedWeightsUpwind": "Upwind (Prec.)",
    "PrecalculatedWeightsCDF": "Linear (Prec.)",
    "DynamicUpwind": "Upwind (Dyn.)",
    "DynamicCDF": "Linear (Dyn.)",
    "HardCodedCDF": "Linear (Stat.)",
    "HardCodedUpwind": "Upwind (Stat.)",
    "FusedDivLap": "Fused",
    "Fused": "Fused"
}
T= {
    "PrecalculatedWeightsUpwind": "Precalculated Weights",
    "PrecalculatedWeightsCDF": "Precalculated Weights",
    "DynamicUpwind": "Dynamic",
    "DynamicCDF": "Dynamic",
    "HardCodedCDF": "Hardcoded",
    "HardCodedUpwind": "Hardcoded",
    "FusedDivLap": "Fused",
    "Fused": "Fused"
}
O= sorted(list(M.values()))
df["varDisplay"] = df["variant"].apply(lambda x : M[x])
df["varType"] = df["variant"].apply(lambda x : T[x])
sorted_variants = O
def variants(_df, name, plotname):
    sb.set_style("whitegrid")
    sb.color_palette("deep", 8)
    with sb.plotting_context("paper", font_scale=1.7):
        strategies = sorted(_df["strategy"].unique()) 
        fig, axes = plt.subplots(1,len(strategies), figsize=(len(strategies)*3.5,6), sharey=True)
        for i, ax in enumerate(axes.flat):
            strat = strategies[i]
            case_df = _df[_df["strategy"] == strat]
            plot = sb.boxplot(data=case_df, ax=ax, x="varDisplay", y="ms_normed", log_scale=False, hue="varType", legend=False, showfliers=False, order=sorted_variants)
            plot.set_xticklabels(plot.get_xticklabels(), 
                                rotation=45, 
                                horizontalalignment='right'
            )
            plot.set(
                ylabel="Mean Assembly Time Per Cell [ms]",
            )
            plot.set(xlabel=f"{strat.capitalize()}")
            ax.yaxis.set_tick_params(labelleft=True)
            ax.grid(True)
            ax.set_yscale("logit")
        plt.tight_layout()
        plt.savefig(f"variants_{name}.svg")
serial = df[df["Threads"]==1]
parallel = df[df["Threads"]>1]

variants(serial, "serial_cpu", "CPU, Serial")
variants(parallel, "multithread_cpu", "CPU, Parallel")

df.to_csv("out.csv", index=False)