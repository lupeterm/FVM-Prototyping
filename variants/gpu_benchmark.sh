#!/bin/bash
numcases=30
total=$(($numcases*8))
finished=0

for dir in /storage/home/lupeterm/nobackup/julia/workspace/*
do
    #echo "[$finished / $total] Benchmarking ${dir}"   
    if [ ! -d "$dir/case/constant/polyMesh" ]; then
 		echo "$dir/case/constant/polyMesh does not exist."
	 	continue
    fi
    ncells=$(sed -nE 's/.* ([0-9]{2,3}).*/\1/p' $dir/case/system/blockMeshDict)
    ncells=$((ncells * ncells * ncells))
	if grep -Eq "LDC-$ncells" variations_gpu.csv
	then
		echo "[$finished / $total] already calculated for $ncells cells and $nthreads threads"
		finished=$(($finished+1))
		continue
	else
		echo "[$finished / $total] not yet calculated for $ncells cells and $nthreads"
	fi
done

for dir in /storage/home/lupeterm/nobackup/julia/workspace/*
do
	echo "[$finished / $total] Benchmarking ${dir}"   
	if [ ! -d "$dir/case/constant/polyMesh" ]; then
			echo "$dir/case/constant/polyMesh does not exist."
			continue
	fi
	ncells=$(sed -nE 's/.* ([0-9]{2,3}).*/\1/p' $dir/case/system/blockMeshDict)
	ncells=$((ncells * ncells * ncells))
	if [ $ncells -gt 1000 ]; then
		continue	
	fi
	if grep -Eq "LDC-$ncells," variations_gpu.csv
	then
		echo "[$finished / $total] already calculated for $ncells cells and $nthreads threads"
		# finished=$(($finished+1))
		continue
	else
		echo "[$finished / $total] not yet calculated for $ncells cells and $nthreads"
	fi
	
	~/.juliaup/bin/julialauncher -g2 bench_variants_gpu.jl $dir/case
	finished=$(($finished+1))
done
echo "Done with $finished/$total benchmark runs."

