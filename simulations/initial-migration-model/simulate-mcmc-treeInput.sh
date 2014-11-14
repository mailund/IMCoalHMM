#!bin/bash
#SBATCH -c 16
#SBATCH --time=22:00:00
#SBATCH -p normal

echo "fil startet" 

echo "========= Job started  at `date` =========="

echo $SLURM_JOBID

cd /home/svendvn/IMCoalHMM/scripts
cp *py /scratch/$SLURM_JOBID
cd /scratch/$SLURM_JOBID

echo "copied files"


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

#for sim in `eval echo {1..${no_sims}}`; do
    
	num=$RANDOM
	mkdir /home/svendvn/IMCoalHMM-simulations.$num -p
	
    	simdir=/home/svendvn/IMCoalHMM-simulations.$num
    	treefile=${simdir}/trees.nwk

    ms 2 1 -T -r ${coal_rho} ${seg_length} \
		   -I 2 1 1 0.0 -em ${coal_tau_mig} 1 2 ${coal_mig_rate} -em ${coal_tau_mig} 2 1 ${coal_mig_rate} \
		   -ej ${coal_tau_split} 2 1 | tail -n +4 | grep -v // > ${treefile}


	echo "ms run"



        python /home/svendvn/workspace/IMCoalHMM/scripts/initial-migration-model-mcmc.py --treefile $treefile \
	-o INMmcmc-sim-Trees-chain.txt --samples 10000 -k 10 --sd_multiplyer 0.2 --use_trees_as_data

    

#done

cp /scratch/$SLURM_JOBID/*.txt /home/svendvn/

echo "========= Job finished at `date` =========="