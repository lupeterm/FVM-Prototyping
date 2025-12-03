# FVM-Prototyping

**First Milestone:** Translating the [C++ Prototype](https://github.com/chihtaw/FVM-CFD-prototype ) into Julia ✔️

TODOs:
- Refactor Project structure yet again so that the assembly method is plug n' play and can be separately benchmarked ✔️
- stability ✔️ (probably :tm: )



## Lid-Driven Cavity on a 1030301 x 1030301 mesh 
```
BenchmarkTools.Trial: 5 samples with 1 evaluation per sample.
 Range (min … max):  755.584 ms …    2.900 s  ┊ GC (min … max):  0.00% … 73.32%
 Time  (median):        1.552 s               ┊ GC (median):    51.03%
 Time  (mean ± σ):      1.568 s ± 869.129 ms  ┊ GC (mean ± σ):  50.09% ± 33.98%

  █ █                   █      █                              █  
  █▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  756 ms           Histogram: frequency by time           2.9 s <

 Memory estimate: 1.67 GiB, allocs estimate: 49270990.
```

#### Using `julia -p 2` (2 or 8 didnt have a noticeable difference) 

```
BenchmarkTools.Trial: 4 samples with 1 evaluation per sample.
 Range (min … max):  1.150 s …    1.618 s  ┊ GC (min … max): 35.72% … 50.97%
 Time  (median):     1.495 s               ┊ GC (median):    44.07%
 Time  (mean ± σ):   1.440 s ± 211.350 ms  ┊ GC (mean ± σ):  44.34% ±  6.32%

  █                                █                  █    █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁█ ▁
  1.15 s         Histogram: frequency by time         1.62 s <

 Memory estimate: 1.67 GiB, allocs estimate: 49270988.
```

## Lid-Driven Cavity on a 8000000 x 8000000 mesh ##
```
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 16.058 s (45.37% GC) to evaluate,
 with a memory estimate of 13.91 GiB, over 393192487 allocations.
```

#### Using `julia -p 2` (2 or 8 didnt have a noticeable difference) 
```
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 11.712 s (45.17% GC) to evaluate,
 with a memory estimate of 13.91 GiB, over 393192485 allocations.
```
## WindsorBody on a 6517376x6517376 mesh ##
```
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 11.594 s (50.27% GC) to evaluate,
 with a memory estimate of 11.70 GiB, over 302869853 allocations.
```
#### Using `julia -p 2` (2 or 8 didnt have a noticeable difference) 

```
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 10.257 s (47.26% GC) to evaluate,
 with a memory estimate of 11.70 GiB, over 302869851 allocations.
 ```
