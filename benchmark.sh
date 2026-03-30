#!/bin/sh

# julia run_threaded.jl cases/Lid-Driven-Cavities/140/
# julia -t 2 run_threaded.jl cases/Lid-Driven-Cavities/200/
# julia -t 4 run_threaded.jl cases/Lid-Driven-Cavities/200/
# julia -t 8 run_threaded.jl cases/Lid-Driven-Cavities/200/
# julia -t 16 run_threaded.jl cases/Lid-Driven-Cavities/200/
# julia -t 32 run_threaded.jl cases/Lid-Driven-Cavities/200/
# for dir in cases/Lid-Driven-Cavities/*/     # list directories in the form "/tmp/dirname/"
# do
#     echo Benchmarking "${dir}"   
    # julia run_threaded.jl $dir
    # julia -t 2 run_threaded.jl $dir
    # julia -t 4 run_threaded.jl $dir

    # julia -t 8 run_threaded.jl cases/Lid-Driven-Cavities/200/
    # julia -t 16 run_threaded.jl cases/Lid-Driven-Cavities/200/
    # julia -t 32 run_threaded.jl cases/Lid-Driven-Cavities/200/
# done
julia run_threaded.jl cases/Lid-Driven-Cavities/40/
julia -t 2 run_threaded.jl cases/Lid-Driven-Cavities/40/
julia -t 4 run_threaded.jl cases/Lid-Driven-Cavities/40/
julia run_threaded.jl cases/Lid-Driven-Cavities/60/
julia -t 2 run_threaded.jl cases/Lid-Driven-Cavities/60/
julia -t 4 run_threaded.jl cases/Lid-Driven-Cavities/60/