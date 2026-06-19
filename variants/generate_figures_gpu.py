import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import pandas as pd
import seaborn as sb
import seaborn.objects as so

df = pd.read_csv("variations_gpu.csv", skip_blank_lines=True)
df["cells"] = df["case_long"].apply(lambda x: int(x.split("-")[1]))
df["ms_normed"] = df["time_mean_ms"] / df["cells"]
df["ns_normed"] = df["ms_normed"]* 1000000

S = {
    "faceBased": "Face-Based",
    "globalFaceBased": "Face-Based Global",
    "cellBased" : "Cell-Based",
    "batchedFace" : "Face-Based Batched"
}
df["s_display"] = df["strategy"].apply(lambda x : S[x])
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
        plt.savefig(f"variants_{name}_gpu.svg")

def normed(_df):
    # _df["dims"] = _df["cells"].apply(lambda x: int(x**(1/3)))
    # dims = sorted(_df["dims"].unique())
    sb.set_theme(rc={'figure.figsize':(20, 7)})
    sb.set_style("whitegrid")
    sb.color_palette("deep", 8)
    order = sorted(_df["s_display"].unique())
    
    with sb.plotting_context("paper", font_scale=2.3):
        fig, a= plt.subplots(1, 2)
        axes = a.flat

        p = sb.lineplot(
            data=_df,
            x="cells",
            y="time_mean_ms",
            hue="s_display",
            style="s_display",
            markers=True,
            markersize=12,
            hue_order=order,
            dashes=False,
            ax=axes[0]
        )
        p.set(
            ylabel="Assembly Time [ms]",
            xlabel="#Cells",
            yscale="log",
            xscale="log",
            # xticks=dims,
            # xticklabels=sorted(_df["cells"].unique())
            # xscale="log",
        )
        p = sb.lineplot(
            data=_df,
            x="cells",
            y="ns_normed",
            hue="s_display",
            style="s_display",
            hue_order=order,
            markers=True,
            markersize=12,
            dashes=False,
            ax=axes[1]
        )
        p.set(
            ylabel="Assembly Time per Cell [ns]",
            xlabel="#Cells",
            yscale="log",
            xscale="log",
            # xticks=dims,
            # xticklabels=sorted(_df["cells"].unique())
            # xscale="log",
        )
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        # # One shared legend above all subplots, 4 columns
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=4,
            # title="Fused Strategy",
            frameon=False,
        )
        # plt.tight_layout()
        fig.subplots_adjust(
            left=0.07,
            right=0.98,
            # bottom=0.18,
            # top=0.78,
            wspace=0.3,
        )
        print(f"writing into normed_gpu.svg")
        plt.savefig(f"subplots_gpu.svg")


normed(df[df["variant"] == "Fused"])

# variants(df, "gpu", "GPU")
batched = df[df["strategy"] == "batchedFace"]
batched = batched[batched["variant"] == "Fused"]
batched.to_csv("batched.csv", index=False)
