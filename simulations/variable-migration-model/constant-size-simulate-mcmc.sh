#!/bin/bash

no_sims=1
no_chunks=1 # FIXME

Ne=20000
gen=25
mu=10^-9
rho_per_gen=$(bc -l <<< 0.01/10^6)
seg_length=1000000
coal_mig_rate=0.5

sym_coal_mig_rate=$(bc -l <<< 2*${coal_mig_rate}) # times two because ms divides by number of populations

theta_years=$(( 4*$Ne*$gen ))
coal_rho=$(bc -l <<< 4*$Ne*$rho_per_gen*$seg_length)

# units in substitutions
theta_subs=$(bc -l <<< $theta_years*$mu )
rho_subs=$(bc -l <<< $rho_per_gen/$gen/$mu )


for sim in `eval echo {1..${no_sims}}`; do
    
	num=$RANDOM
	mkdir /home/svendvn/IMCoalHMM-simulations.$num -p
	
    simdir=/home/svendvn/IMCoalHMM-simulations.$num
    treefile=${simdir}/trees.nwk
    seqfile=${simdir}/alignment.phylip

	echo "directories made"
    
    for chunk in `eval echo {1..${no_chunks}}`; do
        ziphmmfile11=${simdir}/alignment.$chunk.11.ziphmm
        ziphmmfile12=${simdir}/alignment.$chunk.12.ziphmm
        ziphmmfile22=${simdir}/alignment.$chunk.22.ziphmm
	
        ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 ${sym_coal_mig_rate} | tail -n +4 | grep -v // > ${treefile}
        seq-gen -q -mHKY -l ${seg_length} -s ${theta_subs} -p $(( $seg_length / 10 )) < ${treefile} > ${seqfile}
	echo "data made"
        python /home/svendvn/workspace/IMCoalHMM/scripts/prepare-alignments.py --names=1,2 ${seqfile} phylip ${ziphmmfile11} --where_path_ends 3
        python /home/svendvn/workspace/IMCoalHMM/scripts/prepare-alignments.py --names=1,3 ${seqfile} phylip ${ziphmmfile12} --where_path_ends 3
        python /home/svendvn/workspace/IMCoalHMM/scripts/prepare-alignments.py --names=3,4 ${seqfile} phylip ${ziphmmfile22} --where_path_ends 3
	echo "alignments made"
        
    done
    	


	python /home/svendvn/workspace/IMCoalHMM/scripts/variable-migration-model-mcmc.py -o INMmcmc-sim-${i}-chain.txt -a11 ${simdir}/*.11.ziphmm -a12 ${simdir}/*.12.ziphmm -a22 ${simdir}/*.22.ziphmm --samples 10 --thinning 1 --change_often 2 3 6 7 --switch 5 --scew 1
	

done
