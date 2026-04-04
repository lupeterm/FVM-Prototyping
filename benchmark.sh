#!/bin/sh

# julia -t 2 run_threaded.jl cases/Lid-Driven-Cavities/20/
# julia -t 2 run_threaded.jl cases/Lid-Driven-Cavities/20/
for dir in cases/Lid-Driven-Cavities/*/     # list directories in the form "/tmp/dirname/"
do
    echo Benchmarking "${dir}"   
    julia run_threaded.jl      $dir
    julia -t 2 run_threaded.jl $dir
    julia -t 4 run_threaded.jl $dir
    julia -t 8 run_threaded.jl $dir
    julia -t 16 run_threaded.jl $dir
    julia -t 32 run_threaded.jl $dir
    julia -t 64 run_threaded.jl $dir
    julia -t 128 run_threaded.jl $dir
    julia -t 256 run_threaded.jl $dir
done
