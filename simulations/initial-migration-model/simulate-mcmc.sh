#!/bin/bash

split_mya=4
migration_mya=1
coal_mig_rate=0.5

no_sims=1 #no_sims=2
no_chains=1 #no_chains=3
no_pieces=2

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

echo "sim.isolation.period sim.migration.period sim.theta sim.rho sim.mig " > mcmc-simulated-params.txt
echo -e "${subs_isolation_time}\t${subs_migration_time}\t${subs_theta}\t${subs_rho}\t${subs_mig}" >> mcmc-simulated-params.txt

for sim in `eval echo {1..${no_sims}}`; do
    
		mkdir /home/svendvn/IMCoalHMM-simulations.4 -p
	simdir=/home/svendvn/IMCoalHMM-simulations.4
    #simdir=`mktemp -d /tmp/IMCoalHMM-simulations.XXXXXX`
    treefile=${simdir}/trees.nwk
	for piece in `eval echo {1..${no_pieces}}`; do  
		mkdir ${simdir}/${piece} -p  
		seqfile[$piece]=${simdir}/alignment.${piece}.phylip
		ziphmmfile[$piece]=${simdir}/alignment.${piece}.ziphmm
	done
    

	echo "directory made"
    
    ms 2 1 -T -r ${coal_rho} ${seg_length} \
		   -I 2 1 1 0.0 -em ${coal_tau_mig} 1 2 ${coal_mig_rate} -em ${coal_tau_mig} 2 1 ${coal_mig_rate} \
		   -ej ${coal_tau_split} 2 1 | tail -n +4 | grep -v // > ${treefile}


	echo "ms run"

    for piece in `eval echo {1..${no_pieces}}`; do 
	seq-gen -q -mHKY -l ${seg_length} -s ${subs_theta} -p $(( $seg_length / 10 )) < ${treefile} > ${seqfile[$piece]}
	echo "seq-gen run ${piece}"	
	python /home/svendvn/workspace/IMCoalHMM/scripts/prepare-alignments.py ${seqfile[$piece]} phylip ${ziphmmfile[$piece]} --verbose --where_path_ends 3
	echo "zipmm made ${piece}"
    done

	for i in `eval echo {1..2}`; do
        	python /home/svendvn/workspace/IMCoalHMM/scripts/initial-migration-model-mcmc.py ${simdir}/*.ziphmm \
		-o INMmcmc-sim-${i}-chain.txt --samples 4 -k 20 --sd_multiplyer 0.2 --mc3
		echo "finished transform" $i
	done
	echo "now finished " ${chains}
    

done
