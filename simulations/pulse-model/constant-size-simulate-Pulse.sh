#!/bin/bash

source activate coalHMM

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
tmax=$(bc -l <<< 7*$theta_years)

# units in substitutions
theta_subs=$(bc -l <<< $theta_years*$mu )
rho_subs=$(bc -l <<< $rho_per_gen/$gen/$mu )

echo $theta_subs
echo $rho_subs
echo $tmax
echo $(bc -l <<< $coal_mig_rate/$theta_subs)

for sim in `eval echo {1..${no_sims}}`; do
    
	num=$RANDOM
	mkdir /home/svendvn/IMCoalHMM-simulations.$num -p
	
    simdir=/home/svendvn/IMCoalHMM-simulations.$num

	echo "directories made"
    
    for chunk in `eval echo {1..${no_chunks}}`; do
	
        ziphmmfile11=${simdir}/alignment.$chunk.11.ziphmm
        ziphmmfile12=${simdir}/alignment.$chunk.12.ziphmm
        ziphmmfile22=${simdir}/alignment.$chunk.22.ziphmm



	
	cd /home/svendvn/git/IMCoalHMM/simulations/pulse-model
	fsc25 -i parameters3.par -n 1
	cd /home/svendvn
	mv /home/svendvn/git/IMCoalHMM/simulations/pulse-model/parameters3 ${simdir}
	python /home/svendvn/git/IMCoalHMM/scripts/arlequin_to_fasta2.py ${simdir}/parameters3/parameters3_1_1.arp 1000000 ${simdir}
	cat ${simdir}/sample_1_2.fasta >> ${simdir}/sample_1_1.fasta
	cat ${simdir}/sample_2_1.fasta >> ${simdir}/sample_1_1.fasta
	cat $simdir/sample_2_2.fasta >> $simdir/sample_1_1.fasta
	seqfile=${simdir}/sample_1_1.fasta

	echo "data made"
        python /home/svendvn/git/IMCoalHMM/scripts/prepare-alignments.py --names=1,2 ${seqfile} fasta ${ziphmmfile11} --where_path_ends 3
        python /home/svendvn/git/IMCoalHMM/scripts/prepare-alignments.py --names=1,3 ${seqfile} fasta ${ziphmmfile12} --where_path_ends 3
        python /home/svendvn/git/IMCoalHMM/scripts/prepare-alignments.py --names=3,4 ${seqfile} fasta ${ziphmmfile22} --where_path_ends 3
	
	echo "alignments made"
        
    done
    	


	out2="        ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 ${sym_coal_mig_rate} | tail -n +4 | grep -v // > ${treefile}"
	out1="	python /home/svendvn/git/IMCoalHMM/scripts/variable-migration-model-mcmc.py -o INMmcmc-smallVar-sim-${sim}-chain.txt -a11 ${simdir}/*.11.ziphmm -a12 ${simdir}/*.12.ziphmm -a22 ${simdir}/*.22.ziphmm --samples 70  --thinning 1  --sd_multiplyer 0.05 --switch 2 --adap 1 --adap_desired_accept 0.2"
	python /home/svendvn/git/IMCoalHMM/scripts/pulse-model-mcmc.py -o INMmcmc-smallVar-sim-${sim}-chain.txt -a11 ${simdir}/*.11.ziphmm -a12 ${simdir}/*.12.ziphmm -a22 ${simdir}/*.22.ziphmm \
		--samples 2000 --thinning 1  --sd_multiplyer 0.1 --adap 3 --adap_harmonic_power 0.5 --adap3_tracking_begin 10 --adap3_correlates_begin 300 --adap3_from_identical 0.25 --adap3_from_independent 0.25 --mc3 \
		--breakpoints_tail_pieces 4 --intervals 5 5 5 5 --mc3_fixed_temp 20 --mc3_sort_chains --mc3_flip_suggestions 50 --fix_time_points 5 0.0005 10 0.0010 --joint_scaling 0 \
		--parallels 2 --startWithGuessElaborate 1000 1000 1000 1000 1000 1000 1000 1000 0.1 0.1 1.0 0.2 0.2 0.0 0.4 1.0 --fix_params 10 13
	echo $out1 >> INMmcmc-smallVar-sim-${sim}-chain.txt
	echo $out2 >> INMmcmc-smallVar-sim-${sim}-chain.txt
	

done
