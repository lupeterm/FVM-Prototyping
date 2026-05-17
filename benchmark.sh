#!/bin/bash
numcases=$(find ../nobackup/julia/view/base/mesh/ -maxdepth 1 |wc -l)
total=$(($numcases*8))
finished=0
for dir in /storage/home/lupeterm/nobackup/julia/workspace/*
do
    echo "[$finished / $total] Benchmarking ${dir}"   
    if [ ! -d "$dir/case/constant/polyMesh" ]; then
 	 echo "$dir/case/constant/polyMesh does not exist."
	 continue
    fi
    ncells=$(sed -nE 's/.* ([0-9]{2,3}).*/\1/p' $dir/case/system/blockMeshDict)
    ncells=$((ncells * ncells * ncells))
    if [  $ncells !=  64000000 ]; then
    	continue
	fi
	for nthreads in 1 2 4  8 16 32 64 128
    do
    #	if grep -Eq "LDC-$ncells,.+-$nthreads" results/clusterCPUScalingNew.csv
    #	then
	#		echo "[$finished / $total] already calculated for $ncells cells and $nthreads threads"
	#	    finished=$(($finished+1))
	#		continue
	#	else
	#		echo "[$finished / $total] not yet calculated for $ncells cells and $nthreads"
	#	fi
    	    
	 	~/.juliaup/bin/julialauncher -t $nthreads run_threaded.jl $dir
    done
done
echo "Done With benchmark."

