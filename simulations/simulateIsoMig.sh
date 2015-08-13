#!/bin/bash

echo "fil startet" 

echo "========= Job started  at `date` =========="

echo $SLURM_JOBID

cd /home/svendvn/IMCoalHMM/scripts
cp *py /scratch/$SLURM_JOBID
cd /scratch/$SLURM_JOBID

echo "copied files"

filename=$1 #this is the name of the file from which we want to simulate. It is either a isolation file or a migration file
lastpart=echo $filename | cut -d "-" -f 4
lastpart=echo $lastpart | cut -d "." -f 1
firstPart=echo $filename | cut -d "." -f 1



no_sims=25
no_chunks=100 
seg_length=100000

for sim in `eval echo {1..${no_sims}}`; do
    
	num=$RANDOM
		mkdir /scratch/$SLURM_JOBID/IMCoalHMM-simulations.$num -p
	simdir=/scratch/$SLURM_JOBID/IMCoalHMM-simulations.$num
    treefile=${simdir}/trees.nwk
    seqfileUnphased=${simdir}/alignment_unphased.phylip 
    seqfile=${simdir}/alignment.phylip

	echo "directories made"
    
    for chunk in `eval echo {1..${no_chunks}}`; do
        ziphmmfile=${simdir}/alignment.$chunk.ziphmm
	
	python /scratch/$SLURM_JOBID/simulateIsoMig.py --filename=$filename --seqfile=$seqfileUnphased --seg_length=$seg_length --treefile=$treefile
	echo "data made"
	python /scratch/$SLURM_JOBID/phase_the_unphased.py --names=1,2,3,4 --ouput_name=$seqfile --in_filename=$seqfileUnphased
	echo "phasing made"
        python /scratch/$SLURM_JOBID/prepare-alignments.py --names=1,2 ${seqfile} phylip ${ziphmmfile} --where_path_ends 3
       	echo "alignments made"
        
    done
    	
	
	if ["$lastpart"="isolation"];
	then
		python isolation-model.py --header ${simdir}/alignment.*.ziphmm -o $firstPart-SIMULATE-$sim.txt --states 20
	else
		python initial-migration-model.py --header ${simdir}/alignment.*.ziphmm -o $firstPart-SIMULATE-$sim.txt
	fi

	cp /scratch/$SLURM_JOBID/$firstPart-SIMULATE-$sim.txt /home/svendvn/
	
done

echo "========= Job finished at `date` =========="
