#!/bin/bash

split1_mya=4
split2_mya=6
no_sims=10

Ne=20000
gen=25
mu=10^-9
split1_in_years=$(bc -l <<< ${split1_mya}*10^6)
split2_in_years=$(bc -l <<< ${split2_mya}*10^6)
split_diff=$(bc -l <<< ${split2_in_years}-${split1_in_years})
rho_per_gen=$(bc -l <<< 0.01/10^6)
seg_length=1000000

theta_years=$(( 4*$Ne*$gen ))

# recombination and split time in ms units, i.e. units of 4Ne gen
coal_rho=$(bc -l <<< 4*$Ne*$rho_per_gen*$seg_length)
coal_tau1=$(bc -l <<< $split1_in_years/$theta_years )
coal_tau2=$(bc -l <<< $split2_in_years/$theta_years )

# units in substitutions
tau1_subs=$(bc -l <<< $split1_in_years*$mu )
tau2_subs=$(bc -l <<< $split_diff*$mu )
theta_subs=$(bc -l <<< $theta_years*$mu )
rho_subs=$(bc -l <<< $rho_per_gen/$gen/$mu )


#echo ms 3 1 -T -r ${coal_rho} ${seg_length} -I 3 1 1 1 0.0 -ej ${coal_tau1} 2 1 -ej ${coal_tau2} 3 1 
#exit

echo "sim.tau1 sim.tau2 sim.theta sim.rho mle.tau mle.theta mle.rho logL"
for sim in `eval echo {1..${no_sims}}`; do
    
    simdir=`mktemp -d /tmp/IMCoalHMM-simulations.XXXXXX`
    treefile=${simdir}/trees.nwk
    seqfile=${simdir}/alignment.phylip
    ziphmmfile=${simdir}/alignment.ziphmm
    
    #echo ms 3 1 -T -r ${coal_rho} ${seg_length} -I 3 1 1 1 0.0 -ej ${coal_tau1} 2 1 -ej ${coal_tau2} 3 1
    #echo $theta_years ${tau1_subs} ${tau2_subs} ${theta_subs} ${rho_subs}
    #0.004 0.006       0.00640547  0.00709861 |-> 2 3 3.202735 3.549305
    #(for _ in {1..10}; do 
    #    ms 3 1 -T -r ${coal_rho} ${seg_length} -I 3 1 1 1 0.0 -ej ${coal_tau1} 2 1 -ej ${coal_tau2} 3 1  | tail +4 | grep -v //
    #done) | python count-topologies.py 2 3 3.202735 3.549305
    #exit


    ms 3 1 -T -r ${coal_rho} ${seg_length} -I 3 1 1 1 0.0 -ej ${coal_tau1} 2 1 -ej ${coal_tau2} 3 1  | tail +4 | grep -v // > ${treefile}
    seq-gen -q -mHKY -l ${seg_length} -s ${theta_subs} -p $(( $seg_length / 10 )) < ${treefile} > ${seqfile}

    
    ../../scripts/prepare-alignments.py --names=1,2,3 ${seqfile} phylip ${ziphmmfile}

	echo -ne "${tau1_subs}\t${tau2_subs}\t${theta_subs}\t${rho_subs}\t"
	../../scripts/ils-isolation-model.py --logfile=/dev/stdout ${ziphmmfile}
	#for optimizer in Nelder-Mead Powell L-BFGS-B TNC; do
	#	
    #	echo -ne "${tau_subs}\t${theta_subs}\t${rho_subs}\t${optimizer}\t"
    #	isolation-model.py --optimizer=${optimizer} ${ziphmmfile}
    #
	#done
	
    rm ${treefile}
    rm ${seqfile}
    rm -rf ${ziphmmfile}/nStates2seq/
    rm ${ziphmmfile}/*
    rmdir ${ziphmmfile}
    rmdir ${simdir}
done
