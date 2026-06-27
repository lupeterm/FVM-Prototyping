#set table(
  stroke: (x, y) => {
    if y == 0 {
      (top: 1pt + black)
    } else if y == 9 {
      (bottom: 0.9pt + black)
    } else {
      none
    }
  },
)

#let best(body) = strong[#body]

#let drivaer-performance-table = figure(
  table(
    columns: (1.35fr, 0.6fr, 0.95fr, 1fr, 1.2fr, 0.8fr),
    align: (center, center, center, center, center, center),
    inset: (x: 5pt, y: 4pt),
    table.header(
      [Strategy],
      [Impl.],
      [Avg. time (us)],
      [Bandwidth (GB/s)],
      [Speedup vs. NeoN],
      [COE],
    ),
    [Cell-Based], [JuNe], [51,448], [208.54], [2.57x], [1,095.4],
    [Cell-Based], [NeoN], [132,103], [81.22], [1.00x], [426.6],
    [Fused Cell-Based], [NeoN], [480,995], [22.95], [0.27x], [117.2],
    [Face-Based], [JuNe], best[49,677], best[215.98], best[4.68x], best[1,134.5],
    [Face-Based], [NeoN], [232,442], [46.16], [1.00x], [242.5],
    [Fused Face-Based], [NeoN], [n/a], [n/a], [n/a], [n/a],
    [Global Face-Based], [JuNe], [52,118], [205.86], [3.96x], [1,081.4],
    [Global Face-Based], [NeoN], [206,150], [53.56], [1.00x], [273.4],
  ),
  caption: [
    DrivAer GPU benchmark on H200. Measured values are means over three runs.
    Bandwidth follows `nnz * 24 / (time_us / 1e6) / 1e9`.
    COE uses `FVOPS = ncells / time` and
    `COE = FVOPS_gpu / (FVOPS_cpu / numcores)` against the icoFoam H200 DrivAer
    row with #raw("nprocs = 64") from #raw("results/icofoam-h200.csv").
    Speedups compare only compatible traversal families:
    cell-based and fused cell-based against NeoN Cell-Based,
    face-based and fused face-based against NeoN Face-Based,
    and global face-based against NeoN Global Face-Based.
    The NeoN Fused Face-Based row is listed explicitly; #raw("results/drivaer.csv")
    currently contains no DrivAer measurement for it.
  ],
)

#drivaer-performance-table
