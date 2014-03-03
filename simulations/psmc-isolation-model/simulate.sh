#!/bin/bash

no_sims=1

Ne=20000
gen=25
mu=10^-9
rho_per_gen=$(bc -l <<< 0.01/10^6)
seg_length=1000000

theta_years=$(( 4*$Ne*$gen ))
coal_rho=$(bc -l <<< 4*$Ne*$rho_per_gen*$seg_length)

# units in substitutions
theta_subs=$(bc -l <<< $theta_years*$mu )
rho_subs=$(bc -l <<< $rho_per_gen/$gen/$mu )


for sim in `eval echo {1..${no_sims}}`; do
    
    simdir=`mktemp -d /tmp/IMCoalHMM-simulations.XXXXXX`
    treefile=${simdir}/trees.nwk
    seqfile=${simdir}/alignment.phylip
    ziphmmfile=${simdir}/alignment.ziphmm
    
    ms 2 1 -T -r ${coal_rho} ${seg_length} | tail +4 | grep -v // > ${treefile}
    seq-gen -q -mHKY -l ${seg_length} -s ${theta_subs} -p $(( $seg_length / 10 )) < ${treefile} > ${seqfile}

    prepare-alignments.py ${seqfile} phylip ${ziphmmfile}

    psmc-isolation-model.py ${ziphmmfile}
    
    rm ${treefile}
    rm ${seqfile}
    rm ${ziphmmfile}/nStates2seq/*
    rmdir ${ziphmmfile}/nStates2seq
    rm ${ziphmmfile}/*
    rmdir ${ziphmmfile}
    rmdir ${simdir}
done