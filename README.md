# FVM-Prototyping

**First Milestone:** Translating the [C++ Prototype](https://github.com/chihtaw/FVM-CFD-prototype ) into Julia ✔️

TODOs:
- Refactor Project structure yet again so that the assembly method is plug n' play and can be separately benchmarked ✔️
- stability ✔️ (probably :tm: )



###### Lid-Driven Cavity on a 1030301 x 1030301 mesh ######

Reading Points..
Reading Owners..
Reading Faces..
Reading Neighbors..
Reading Boundaries..
Constructing Cells..
Reading /home/peter/clones/cases/ldc/Lid_driven_cavity-3d/S/constant/transportProperties
Variable: RegexMatch("nu              [0 2 -1 0 0 0 0] 0.01;", 1="0.01")


--> Using VectorAssembly <--

BenchmarkTools.Trial: 5 samples with 1 evaluation per sample.
 Range (min … max):  755.584 ms …    2.900 s  ┊ GC (min … max):  0.00% … 73.32%
 Time  (median):        1.552 s               ┊ GC (median):    51.03%
 Time  (mean ± σ):      1.568 s ± 869.129 ms  ┊ GC (mean ± σ):  50.09% ± 33.98%

  █ █                   █      █                              █  
  █▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  756 ms           Histogram: frequency by time           2.9 s <

 Memory estimate: 1.67 GiB, allocs estimate: 49270990.



##############################################################################################

###### Lid-Driven Cavity on a 8000000 x 8000000 mesh ######

Reading Points..
Reading Owners..
Reading Faces..
Reading Neighbors..
Reading Boundaries..
Constructing Cells..
Reading /home/peter/clones/cases/ldc2/Lid_driven_cavity-3d/M/constant/transportProperties
Variable: RegexMatch("nu              [0 2 -1 0 0 0 0] 0.01;", 1="0.01")


--> Using VectorAssembly <--

BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 16.058 s (45.37% GC) to evaluate,
 with a memory estimate of 13.91 GiB, over 393192487 allocations.



##############################################################################################

###### WindsorBody on a 6517376x6517376 mesh ######

Reading Points..
Reading Owners..
Reading Faces..
Reading Neighbors..
Reading Boundaries..
Constructing Cells..
Reading /home/peter/clones/cases/windsor/workspace/0172cca15b84871d59583e93ed420b34/case/constant/transportProperties
Variable: RegexMatch("nu              [0 2 -1 0 0 0 0] 1.44e-05;", 1="1.44e-05")


--> Using VectorAssembly <--

BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 11.594 s (50.27% GC) to evaluate,
 with a memory estimate of 11.70 GiB, over 302869853 allocations.



##############################################################################################