#!/bin/bash

split_mya=2
no_sims=10

Ne=20000
gen=25
mu=10^-9
split_in_years=$(bc -l <<< ${split_mya}*10^6)
rho_per_gen=$(bc -l <<< 0.01/10^6)
seg_length=1000000

theta_years=$(( 4*$Ne*$gen ))

# recombination and split time in ms units, i.e. units of 4Ne gen
coal_rho=$(bc -l <<< 4*$Ne*$rho_per_gen*$seg_length)
coal_tau=$(bc -l <<< $split_in_years/$theta_years )

# units in substitutions
tau_subs=$(bc -l <<< $split_in_years*$mu )
theta_subs=$(bc -l <<< $theta_years*$mu )
rho_subs=$(bc -l <<< $rho_per_gen/$gen/$mu )


echo "sim.tau sim.theta sim.rho mle.tau mle.theta mle.rho logL"
for sim in `eval echo {1..${no_sims}}`; do
    
    simdir=`mktemp -d /tmp/IMCoalHMM-simulations.XXXXXX`
    treefile=${simdir}/trees.nwk
    seqfile=${simdir}/alignment.phylip
    ziphmmfile=${simdir}/alignment.ziphmm
    
    ms 2 1 -T -r ${coal_rho} ${seg_length} -I 2 1 1 0.0 -ej ${coal_tau} 2 1 | tail +4 | grep -v // > ${treefile}
    seq-gen -q -mHKY -l ${seg_length} -s ${theta_subs} -p $(( $seg_length / 10 )) < ${treefile} > ${seqfile}

    prepare-alignments.py ${seqfile} phylip ${ziphmmfile}

	echo -ne "${tau_subs}\t${theta_subs}\t${rho_subs}\t"
	isolation-model.py ${ziphmmfile}
	#for optimizer in Nelder-Mead Powell L-BFGS-B TNC; do
	#	
    #	echo -ne "${tau_subs}\t${theta_subs}\t${rho_subs}\t${optimizer}\t"
    #	isolation-model.py --optimizer=${optimizer} ${ziphmmfile}
    #
	#done
	
    rm ${treefile}
    rm ${seqfile}
    #rm -rf ${ziphmmfile}/nStates2seq/
    #rm ${ziphmmfile}/*
    #rmdir ${ziphmmfile}
    rm ${ziphmmfile}
    rmdir ${simdir}
done