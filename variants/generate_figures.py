import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter
import pandas as pd
import seaborn as sb
import seaborn.objects as so

# df = pd.read_csv("variations_cpu_polyester_pinnedthreads.csv", skip_blank_lines=True)
df = pd.read_csv("variations_cpu_polyester.csv", skip_blank_lines=True)
df["cells"] = df["case_long"].apply(lambda x: int(x.split("-")[1]))
df["ms_normed"] = df["time_mean_ms"] / df["cells"]
S = {
    "faceBased": "Face-based",
    "globalFaceBased": "Global Face-based",
    "cellBased" : "Cell-based",
    "batchedFace" : "Batched Face-based"
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
    with sb.plotting_context("paper", font_scale=2):
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
            plot.set(xlabel=S[strat])
            ax.yaxis.set_tick_params(labelleft=True)
            ax.grid(True)
            ax.set_yscale("logit")
        plt.tight_layout()
        plt.savefig(f"variants_{name}_poly.svg")
    plt.clf()
    plt.cla()
    plt.close()


# def speedup(_df):
#     _df = _df[_df["cells"] == _df["cells"].max()]

#     group_cols = [
#         "case",
#         "strategy",
#     ]
#     baseline = (
#         _df[_df["Threads"] == 1]
#         .groupby(group_cols, as_index=False)["time_mean_ms"]
#         .first()
#         .rename(columns={"time_mean_ms": "serial_time"})
#     )
#     _df = _df.merge(baseline, on=group_cols, how="left")
#     _df["speedup"] = _df["serial_time"] / _df["time_mean_ms"]
#     _df["efficiency"] = _df["speedup"] / _df["Threads"]
#     for strat in _df["s_display"].unique():
#         med = _df[(_df["s_display"] == strat) & (_df["Threads"] == 128)]["speedup"].median()
#         print(f"{strat}: {med}")
#     sb.set_theme(rc={'figure.figsize':(10, 5)})
#     sb.set_style("whitegrid")
#     sb.color_palette("deep", 8)
#     ts = _df["Threads"].unique()
    # with sb.plotting_context("paper", font_scale=2):
        
    #     p = sb.lineplot(
    #         data=_df,
    #         x="Threads",
    #         y="speedup",
    #         hue="s_display",
    #         style="s_display",
    #         markers=True,
    #         markersize=12,
    #         dashes=False,
    #         ax=ax
    #     )
    #     ax = p.axes
    #     p.legend(title="Fused Strategy")
    #     p.set(
    #         ylabel="Speedup",
    #         xlabel="#Threads",
    #         xscale="log",
    #         xticks=ts,
    #         xticklabels=[str(t) for t in ts],
    #     )
    #     plt.tight_layout()
    #     print(f"writing into variants_speedup.svg")
    #     # plt.savefig("variants_speedup.svg")
    #     # plt.clf()
    #     # plt.cla()
    #     # plt.close()

def efficiency(_df, ax):
    _df = _df[_df["cells"] == _df["cells"].max()]

    group_cols = [
        "case",
        "strategy",
    ]
    baseline = (
        _df[_df["Threads"] == 1]
        .groupby(group_cols, as_index=False)["time_mean_ms"]
        .first()
        .rename(columns={"time_mean_ms": "serial_time"})
    )
    _df = _df.merge(baseline, on=group_cols, how="left")
    _df["speedup"] = _df["serial_time"] / _df["time_mean_ms"]
    _df["efficiency"] = _df["speedup"] / _df["Threads"]
    print(df["strategy"].unique())
    sb.set_theme(rc={'figure.figsize':(10, 5)})
    sb.set_style("whitegrid")
    sb.color_palette("deep", 8)
    ts = _df["Threads"].unique()
    with sb.plotting_context("paper", font_scale=2):
        
        p = sb.lineplot(
            data=_df,
            x="Threads",
            y="efficiency",
            err_style=None,
            hue="s_display",
            style="s_display",
            markers=True,
            markersize=12,
            dashes=False,
            ax=ax
        )
        p.legend(title="Fused Strategy")
        p.set(
            ylabel="Efficiency",
            xlabel="#Threads",
            xscale="log",
            xticks=ts,
            xticklabels=[str(t) for t in ts],
        )
        plt.tight_layout()
        print(f"writing into variants_efficiency.svg")
    #     plt.savefig("variants_efficiency.svg")
    # plt.clf()
    # plt.cla()
    # plt.close()


def time(_df, ax):
    _df = _df[_df["cells"] == _df["cells"].max()]
    sb.set_theme(rc={'figure.figsize':(10, 5)})
    sb.set_style("whitegrid")
    sb.color_palette("deep", 8)
    # _df["dims"] = _df["cells"].apply(lambda x: int(x**(1/3)))
    # dims = sorted(_df["dims"].unique())
    with sb.plotting_context("paper", font_scale=1.7):
        p = sb.lineplot(
            data=_df,
            x="Threads",
            y="time_mean_ms",
            hue="s_display",
            style="s_display",
            markers=True,
            markersize=12,
            dashes=False,
            ax=ax
        )

        p.legend(title="Fused Strategy")
        p.set(
            ylabel="Assembly Time per Cell [ms]",
            xlabel="#Threads",
            yscale="log",
            # xticks=dims,
            # xticklabels=sorted(_df["cells"].unique())
            # xscale="log",
        )
        plt.tight_layout()
        print(f"writing into variants_time.svg")
    # plt.savefig(f"variants_time.svg")
    # plt.clf()
    # plt.cla()
    # plt.close()
def normed(_df, ax):
    _df = _df[_df["cells"] == _df["cells"].max()]
    sb.set_theme(rc={'figure.figsize':(10, 5)})
    sb.set_style("whitegrid")
    sb.color_palette("deep", 8)
    # _df["dims"] = _df["cells"].apply(lambda x: int(x**(1/3)))
    # dims = sorted(_df["dims"].unique())
    with sb.plotting_context("paper", font_scale=1.7):
        p = sb.lineplot(
            data=_df,
            x="Threads",
            y="time_mean_ms",
            hue="s_display",
            style="s_display",
            markers=True,
            markersize=12,
            dashes=False,
            ax=ax
        )

        p.legend(title="Fused Strategy")
        p.set(
            ylabel="Assembly Time per Cell [ms]",
            xlabel="#Threads",
            yscale="log",
            # xticks=dims,
            # xticklabels=sorted(_df["cells"].unique())
            # xscale="log",
        )
        plt.tight_layout()
        print(f"writing into variants_time.svg")

def all(_df):
    group_cols = [
        "case",
        "strategy",
    ]
    baseline = (
        _df[_df["Threads"] == 1]
        .groupby(group_cols, as_index=False)["time_mean_ms"]
        .first()
        .rename(columns={"time_mean_ms": "serial_time"})
    )
    _df = _df.merge(baseline, on=group_cols, how="left")
    _df["speedup"] = _df["serial_time"] / _df["time_mean_ms"]
    _df["efficiency"] = _df["speedup"] / _df["Threads"]
    sb.set_theme(rc={'figure.figsize':(20, 7)})
    sb.set_style("whitegrid")
    sb.color_palette("deep", 8)
    ts = _df["Threads"].unique()
    markersize = 14
    order = sorted(_df["s_display"].unique())
    _df["dims"] = _df["cells"].apply(lambda x: int(x**(1/3)))
    dims = sorted(_df["dims"].unique())
    with sb.plotting_context("paper", font_scale=2.3):
        fig, a= plt.subplots(1, 3)
        axes = a.flat
        p = sb.lineplot(
            data=_df[_df["cells"] >= 27000000],
            x="Threads",
            y="efficiency",
            err_style=None,
            hue="s_display",
            style="s_display",
            hue_order=order,
            markers=True,
            markersize=markersize,
            dashes=False,
            ax=axes[0]
        )
        # p.legend(title="Fused Strategy")
        p.set(
            ylabel="Efficiency",
            xlabel="#Threads",
            xscale="log",
            xticks=ts,
            xticklabels=[str(t) for t in ts],
        )

        p = sb.lineplot(
            data=_df[_df["cells"] >= 27000000],
            x="Threads",
            y="speedup",
            hue="s_display",
            hue_order=order,
            err_style=None,
            style="s_display",
            markers=True,
            markersize=markersize,
            dashes=False,
            ax=axes[1]
        )
        ax = p.axes
        axes[1].plot([1, 128], [1, 128], "k--", label="Ideal Speedup")
        # axes[1].set_yscale("log")
        # p.legend(title="Fused Strategy")
        p.set(
            ylabel="Speedup",
            xlabel="#Threads",
            xscale="log",
            yscale="log",
            xticks=ts,
            xticklabels=[str(t) for t in ts],
            yticks=ts,
            yticklabels=[str(t) for t in ts],
        )

        _df["ns_normed"] = _df["ms_normed"]* 1000000
        p = sb.lineplot(
            data=_df[_df["Threads"] == 128],
            x="dims",
            y="ns_normed",
            hue="s_display",
            style="s_display",
            hue_order=order,
            err_style=None,
            markers=True,
            markersize=10,
            dashes=False,
            ax=axes[2]
        )
        p.set(
            ylabel="Assembly Time per Cell [ns]",
            xlabel="#Cells",
            # xscale="log",
            ylim=[0.5, 8000],
            yscale="log",
            xticklabels=["-100", "0", "$100^3$", "$200^3$", "$300^3$", "$400^3$", "125M"]
            # yticks=[1, 10, 100],
            # xticks=ts,
            # xticklabels=[str(t) for t in ts],
        )

        handles, labels = axes[1].get_legend_handles_labels()
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
            ncol=5,
            # title="Fused Strategy",
            frameon=False,
        )
        # plt.tight_layout()
        fig.subplots_adjust(
            left=0.05,
            right=0.99,
            # bottom=0.18,
            # top=0.78,
            wspace=0.25,
        )
        plt.savefig(f"fused_all.svg")
        plt.clf()
        plt.cla()
        plt.close()

serial = df[df["Threads"]==1]
parallel = df[df["Threads"]>1]

# variants(serial, "serial_cpu", "CPU, Serial")
# variants(parallel, "multithread_cpu", "CPU, Parallel")
def speedup(_df):
    group_cols = [
        "case",
        "strategy",
    ]
    baseline = (
        _df[_df["Threads"] == 1]
        .groupby(group_cols, as_index=False)["time_mean_ms"]
        .first()
        .rename(columns={"time_mean_ms": "serial_time"})
    )
    _df = _df.merge(baseline, on=group_cols, how="left")
    _df["speedup"] = _df["serial_time"] / _df["time_mean_ms"]
    _df["efficiency"] = _df["speedup"] / _df["Threads"]
    sb.set_theme(rc={'figure.figsize':(20, 7)})
    sb.set_style("whitegrid")
    sb.color_palette("deep", 8)
    ts = _df["Threads"].unique()
    markersize = 14
    order = sorted(_df["s_display"].unique())
    _df["dims"] = _df["cells"].apply(lambda x: int(x**(1/3)))
    dims = sorted(_df["dims"].unique())
    for strat in _df["s_display"].unique():
        # med = _df[(_df["s_display"] == strat) & (_df["Threads"] == 128) & (~_df["speedup"].isna())]["speedup"]
        print(f"{strat}: {_df['speedup']}")
# speedup(df[df["variant"] == "Fused"])
# time(df[df["variant"] == "Fused"])
# efficiency(df[df["variant"] == "Fused"])
# df.to_csv("out.csv", index=False)
# serial.to_csv("serial.csv", index=False)
# all(df)


def all2(_df):
    group_cols = [
        "case",
        "strategy",
    ]
    baseline = (
        _df[_df["Threads"] == 1]
        .groupby(group_cols, as_index=False)["time_mean_ms"]
        .first()
        .rename(columns={"time_mean_ms": "serial_time"})
    )
    _df = _df.merge(baseline, on=group_cols, how="left")
    _df["speedup"] = _df["serial_time"] / _df["time_mean_ms"]
    _df["efficiency"] = _df["speedup"] / _df["Threads"]
    sb.set_theme(rc={'figure.figsize':(25, 12)})
    sb.set_style("whitegrid")
    sb.color_palette("deep", 8)
    _df["s_t"]  = _df["Threads"].apply(lambda x: str(x))
    ts = _df["Threads"].unique()
    markersize = 14
    order = sorted(_df["s_display"].unique())
    _df["dims"] = _df["cells"].apply(lambda x: int(x**(1/3)))
    dims = sorted(_df["dims"].unique())
    with sb.plotting_context("paper", font_scale=2.3):
        # fig, a= plt.subplots(4)
        _df["ns_normed"] = _df["ms_normed"]* 1000000
        p = sb.relplot(
            data=_df,
            x ="dims",
            y="ns_normed",
            col = "strategy",
            col_wrap=2,
            hue="s_t",
            markers=True,
            style="s_t",
            kind="line"
        )
        # p = sb.lineplot(
        #     data=_df[_df["strategy"] == "globalFaceBased"],
        #     x="dims",
        #     y="ns_normed",
        #     hue="s_t",
        #     style="s_t",
        #     # hue_order=order,
        #     err_style=None,
        #     markers=True,
        #     markersize=13,
        #     dashes=False,
        #     ax=a
        # )
        # axes = p.axes
        # p.legend(title="Global Strategy #Threads")

        p.set(
            ylabel="Assembly Time per Cell [ns]",
            xlabel="#Cells",
            xscale="log",
            # ylim=[0.5, 8000],
            yscale="log",
            xticklabels=["-100", "0", "$100^3$", "$200^3$", "$300^3$", "$400^3$", "125M"]
            # xticklabels=[f"{int(int(x)**3)}" for x in axes[2].get_xticks()]
            # yticks=[1, 10, 100],
            # xticks=ts,
            # xticklabels=[str(t) for t in ts],
        )
        # handles, labels = a.get_legend_handles_labels()
        # leg = a.get_legend()
        # if leg is not None:
        #     leg.remove()
        # fig.legend(
        #     handles,
        #     labels,
        #     loc="center right",
        #     bbox_to_anchor=(1, 0.5),
        #     ncol=1,
        #     # title="Fused Strategy",
        #     frameon=False,
        # )
        # fig.subplots_adjust(
        #     left=0.05,
        #     right=0.99,
        #     # bottom=0.18,
        #     # top=0.78,
        #     wspace=0.25,
        # )
        # # One shared legend above all subplots, 4 columns
        # 
        plt.savefig(f"global.svg")
        plt.clf()
        plt.cla()
        plt.close()
all2(df)
