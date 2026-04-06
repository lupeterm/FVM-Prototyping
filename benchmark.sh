#!/bin/sh

julia run_gpu.jl cases/Lid-Driven-Cavities/20/ 32
# julia run_gpu.jl cases/Wind/ 32
# julia run_gpu.jl cases/Wind/ 64
# julia run_gpu.jl cases/Wind/ 128
# julia run_gpu.jl cases/Wind/ 256
# julia run_gpu.jl cases/Wind/ 512
# julia run_gpu.jl cases/Wind/ 1024
# julia run_gpu.jl cases/Lid-Driven-Cavities/20/
# julia run_gpu.jl cases/Lid-Driven-Cavities/40/
# julia run_gpu.jl cases/Lid-Driven-Cavities/60/
# julia run_gpu.jl cases/Lid-Driven-Cavities/80/
# julia run_gpu.jl cases/Lid-Driven-Cavities/100/
# julia run_gpu.jl cases/Lid-Driven-Cavities/120/
# julia run_gpu.jl cases/Lid-Driven-Cavities/140/
# julia run_gpu.jl cases/Lid-Driven-Cavities/160/
# julia run_gpu.jl cases/Lid-Driven-Cavities/180/
# julia run_gpu.jl cases/Lid-Driven-Cavities/200/
# julia run_gpu.jl cases/Lid-Driven-Cavities/20/
# julia run_gpu.jl cases/Lid-Driven-Cavities/20/
# julia run_gpu.jl cases/Lid-Driven-Cavities/20/
# julia -t 2 run_threaded.jl cases/Lid-Driven-Cavities/20/
# for dir in cases/Lid-Driven-Cavities/*/     # list directories in the form "/tmp/dirname/"
# do
#     echo Benchmarking "${dir}"   
#     julia run_gpu.jl $dir
    # julia run_threaded.jl      $dir
    # julia -t 2 run_threaded.jl $dir
    # julia -t 4 run_threaded.jl $dir
    # julia -t 8 run_threaded.jl $dir
    # julia -t 16 run_threaded.jl $dir
    # julia -t 32 run_threaded.jl $dir
    # julia -t 64 run_threaded.jl $dir
    # julia -t 128 run_threaded.jl $dir
    # julia -t 256 run_threaded.jl $dir
# done
