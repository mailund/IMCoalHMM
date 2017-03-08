#!/bin/bash

split_mya=$1
migration_mya=$2
coal_mig_rate=$3

no_sims=10

Ne=20000
gen=25
mu=10^-9

tau_split_in_years=$(bc -l <<< ${split_mya}*10^6)
tau_mig_in_years=$(bc -l <<< ${migration_mya}*10^6)

isolation_time_in_years=${tau_mig_in_years}
migration_time_in_years=$(bc -l <<< ${tau_split_in_years}-${tau_mig_in_years})
rho_per_gen=$(bc -l <<< 0.01/10^6)
seg_length=1000000

theta_years=$(( 4*$Ne*$gen ))

# recombination and split time in ms units, i.e. units of 4Ne gen
coal_tau_mig=$(bc -l <<< ${tau_mig_in_years}/${theta_years})
coal_tau_split=$(bc -l <<< ${tau_split_in_years}/${theta_years})
coal_rho=$(bc -l <<< 4*${Ne}*${rho_per_gen}*${seg_length})

# units in substitutions
subs_isolation_time=$(bc -l <<< ${isolation_time_in_years}*${mu})
subs_migration_time=$(bc -l <<< ${migration_time_in_years}*${mu})
subs_theta=$(bc -l <<< ${theta_years}*${mu})
subs_rho=$(bc -l <<< ${rho_per_gen}/${gen}/${mu})
subs_mig=$(bc -l <<< ${coal_mig_rate}/${subs_theta})

echo -n "sim.isolation.period sim.migration.period sim.theta sim.rho sim.mig optimizer "
echo    "mle.isolation.period mle.migration.period mle.theta mle.rho mle.mig logL"
for sim in `eval echo {1..${no_sims}}`; do
    
    simdir=`mktemp -d /tmp/IMCoalHMM-simulations.XXXXXX`
    treefile=${simdir}/trees.nwk
    seqfile=${simdir}/alignment.phylip
    ziphmmfile=${simdir}/alignment.ziphmm
    
    ms 2 1 -T -r ${coal_rho} ${seg_length} \
		   -I 2 1 1 0.0 -em ${coal_tau_mig} 1 2 ${coal_mig_rate} -em ${coal_tau_mig} 2 1 ${coal_mig_rate} \
		   -ej ${coal_tau_split} 2 1 | tail +4 | grep -v // > ${treefile}
	
    seq-gen -q -mHKY -l ${seg_length} -s ${subs_theta} -p $(( $seg_length / 10 )) < ${treefile} > ${seqfile}

    prepare-alignments.py ${seqfile} phylip ${ziphmmfile}

	for optimizer in Nelder-Mead Powell L-BFGS-B TNC; do
		
	    echo -ne "${subs_isolation_time}\t${subs_migration_time}\t${subs_theta}\t${subs_rho}\t${subs_mig}\t${optimizer}\t"
	    initial-migration-model.py --optimizer=${optimizer} ${ziphmmfile}
    
	done
	
    
    rm ${treefile}
    rm ${seqfile}
    rm ${ziphmmfile}
    rmdir ${simdir}
done