#set page(paper: "a4", flipped: true, margin: 1.4cm)
#set text(size: 9pt)
#set table(
  stroke: (x, y) => {
    if y == 0 {
      (top: 1pt + black)
    } else if y == 7 {
      (bottom: 0.9pt + black)
    } else {
      none
    }
  },
)

#let muted = rgb("#667085")

#let best(body) = strong[#body]

= Coefficient of Equivalence: icoFoam H200 MPI vs. JuNe GPU

#text(fill: muted)[
  COE is computed as $"COE" = (T_"cpu" dot N_"cores") / (T_"gpu" dot N_"gpus")$.
  Here, #raw("N_gpus = 1") and #raw("N_cores = nprocs") from the
  icoFoam MPI rows. CPU cells are read from #raw("results/icofoam-h200.csv"),
  and JuNe GPU times are averaged from #raw("results/NeoN-GPU.csv") for
  #raw("gpu-nvidia-h200"). The table reports median COE for #raw("nprocs = 1 and 64")
  over all 30 matching cell counts (#raw("10^3") to #raw("400^3")).
]

#v(0.5cm)

#figure(
  table(
    columns: (1.25fr, 1fr, 1fr, 1fr),
    align: (center, center, center, center),
    inset: (x: 5pt, y: 4pt),
    table.header(
      [JuNe strategy],
      [CPU MPI ranks],
      [GPUs],
      [Median COE],
    ),
    [Cell-Based],
    [1],
    [1],
    [55.1],

    [Face-Based],
    [1],
    [1],
    best[97.2],

    [Global Face-Based],
    [1],
    [1],
    [93.7],

    [Cell-Based],
    [64],
    [1],
    [88.3],

    [Face-Based],
    [64],
    [1],
    best[149.2],

    [Global Face-Based],
    [64],
    [1],
    [143.5],
  ),
  caption: [
    Median COE for one JuNe GPU compared against icoFoam H200. Highlighted cells mark
    the best JuNe strategy within each MPI-rank group.
  ],
)
