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
	
        ziphmmfile1A=${simdir}/alignment.$chunk.1A.ziphmm
        ziphmmfile1B=${simdir}/alignment.$chunk.1B.ziphmm
        ziphmmfile2A=${simdir}/alignment.$chunk.2A.ziphmm
        ziphmmfile2B=${simdir}/alignment.$chunk.2B.ziphmm


	
	cd /home/svendvn/git/IMCoalHMM/simulations/admix-3-haplos
	fsc25 -i parameters.par -n 1
	cd /home/svendvn
	mv /home/svendvn/git/IMCoalHMM/simulations/admix-3-haplos/parameters ${simdir}
	python /home/svendvn/git/IMCoalHMM/simulations/admix-3-haplos/arlequin_to_fasta2.py ${simdir}/parameters/parameters_1_1.arp ${seg_length} ${simdir}
	cat ${simdir}/sample_1_2.fasta >> ${simdir}/sample_1_1.fasta
	cat ${simdir}/sample_2_1.fasta >> ${simdir}/sample_1_1.fasta
	cat ${simdir}/sample_2_2.fasta >> ${simdir}/sample_1_1.fasta
	seqfile=${simdir}/sample_1_1.fasta

	echo "data made"
        python /home/svendvn/git/IMCoalHMM/scripts/prepare-alignments.py --names=1,3,4 ${seqfile} fasta ${ziphmmfile1A} --where_path_ends 3
        python /home/svendvn/git/IMCoalHMM/scripts/prepare-alignments.py --names=2,3,4 ${seqfile} fasta ${ziphmmfile1B} --where_path_ends 3
        python /home/svendvn/git/IMCoalHMM/scripts/prepare-alignments.py --names=1,2,3 ${seqfile} fasta ${ziphmmfile2A} --where_path_ends 3
        python /home/svendvn/git/IMCoalHMM/scripts/prepare-alignments.py --names=1,2,4 ${seqfile} fasta ${ziphmmfile2B} --where_path_ends 3
	
	echo "alignments made"
        
    done
    	


	out2="        ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 ${sym_coal_mig_rate} | tail -n +4 | grep -v // > ${treefile}"
	out1="	python /home/svendvn/git/IMCoalHMM/scripts/variable-migration-model-mcmc.py -o INMmcmc-smallVar-sim-${sim}-chain.txt -a11 ${simdir}/*.11.ziphmm -a12 ${simdir}/*.12.ziphmm -a22 ${simdir}/*.22.ziphmm --samples 70  --thinning 1  --sd_multiplyer 0.05 --switch 2 --adap 1 --adap_desired_accept 0.2"
	python /home/svendvn/git/IMCoalHMM/scripts/admix-mediumstep.py -o INMmcmc-smallVar-sim-${sim}-chain.txt --alignments1A ${simdir}/*.1A.ziphmm --alignments1B ${simdir}/*.1B.ziphmm --alignments2A ${simdir}/*.2A.ziphmm  --alignments2B ${simdir}/*.2B.ziphmm --parallel 
	echo $out1 >> INMmcmc-smallVar-sim-${sim}-chain.txt
	echo $out2 >> INMmcmc-smallVar-sim-${sim}-chain.txt
	

done
